# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Controlled position sweep; no production kernel changes."""

import argparse
import hashlib
import json
import random
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention
from aspec.models.ops.cudnn_gqa_backward import _BackwardPlan

LENGTHS = [4096] * 10 + [40960, 49152]
POLICIES = {
    "early_quarter": (0, 0.25),
    "early_half": (0, 0.5),
    "full_random": (0, 1),
    "middle_half": (0.25, 0.75),
    "late_half": (0.5, 1),
    "late_quarter": (0.75, 1),
    "full_even": (0, 1),
}
MODES = ("native", "external", "internal")


def make_inputs(block):
    torch.manual_seed(1019)
    qn = len(LENGTHS) * 512 * block
    kn = sum(LENGTHS) + qn
    q = torch.randn(qn, 32, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k = torch.randn(kn, 8, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    return (q, k, v), torch.randn_like(q)


def make_mask(policy, block, values):
    endpoints, allanchors, docstats = [], [], []
    cstart = 0
    lo, hi = POLICIES[policy]
    for i, n in enumerate(LENGTHS):
        # Common B=7 eligibility also used for B=5: exact same anchors across B.
        low, high = int(n * lo), min(int(n * hi), n - 7)
        gen = torch.Generator().manual_seed(20261031 + i)
        if policy == "full_even":
            positions = torch.linspace(low, high - 1, 512, dtype=torch.float64).round().long()
        else:
            positions = (torch.randperm(high - low, generator=gen)[:512] + low).sort().values
        assert positions.numel() == 512 and positions.unique().numel() == 512
        assert 0 <= positions[0] and positions[-1] < n - 7
        allanchors.append(positions)
        docstats.append(
            {
                "length": n,
                "first": int(positions[0]),
                "last": int(positions[-1]),
                "mean_visible_context": float(positions.double().mean()),
                "mean_position_fraction": float(positions.double().mean() / n),
            }
        )
        a = positions.cuda().repeat_interleave(block)
        begin = torch.arange(512, device="cuda").repeat_interleave(block) * block
        begin += sum(LENGTHS) + i * 512 * block
        left = torch.full_like(a, cstart)
        endpoints.append(torch.stack([torch.zeros_like(a), left, a + cstart, begin, begin + block]))
        cstart += n
    qn, kn = values[0].shape[0], values[1].shape[0]
    mask = CudnnFlexMask(
        torch.cat(endpoints, dim=1)[None].int().contiguous(),
        pack_gqa=None,
        cu_seqlens_q=torch.tensor([0, qn], dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor([0, kn], dtype=torch.int32, device="cuda"),
        max_seqlen_q=qn,
        max_seqlen_k=kn,
    )
    pos = torch.cat(allanchors)
    valid = 32 * block * (int(pos.sum()) + len(LENGTHS) * 512 * block)
    info = {
        "anchor_sha256": hashlib.sha256(pos.numpy().tobytes()).hexdigest(),
        "positions": [x.tolist() for x in allanchors],
        "per_document": docstats,
        "mean_visible_context": float(pos.double().mean()),
        "mean_position_fraction": statistics.mean(x["mean_position_fraction"] for x in docstats),
        "valid_score_pairs_all_heads": valid,
    }
    return mask, info


def tile_counts(mask, values, valid):
    native = mask.plan_for(*values)._runtime_args[0].bwd_tensors
    joint = _BackwardPlan(*values, mask).normalized
    out = {}
    for name, meta, heads in [("native", native, 32), ("joint", joint, 8)]:
        partial, full = int(meta.mask_block_cnt.sum()), int(meta.full_block_cnt.sum())
        size = list(meta.block_size)
        visited = heads * (partial + full)
        slots = visited * size[0] * size[1]
        out[name] = {
            "partial_per_mask": partial,
            "full_per_mask": full,
            "block_size": size,
            "visited_tiles_all_heads": visited,
            "dense_score_slots_all_heads": slots,
            "useful_fraction": valid / slots,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, choices=(5, 7), required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", choices=tuple(POLICIES), default="full_random")
    ap.add_argument("--profile", choices=MODES)
    ap.add_argument("--windows", type=int, default=11)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.windows < 3 or args.repeats < 1:
        ap.error("windows must be >= 3 and repeats must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    if free < 24 * 2**30:
        raise RuntimeError("This 128K benchmark requires at least 24 GiB free device memory")
    values, do = make_inputs(args.block)
    names = [args.policy]
    modes = [args.profile] if args.profile else list(MODES)
    results = {
        "block": args.block,
        "lengths": LENGTHS,
        "anchors_per_doc": 512,
        "total_anchors": 6144,
        "draft_query_rows": len(LENGTHS) * 512 * args.block,
        "q_shape": list(values[0].shape),
        "kv_shape": list(values[1].shape),
        "dtype": "BF16",
        "reduction_dtype": "FP32",
        "pack_gqa": None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Cached-mask CUDA graph attention only, external copies included. Synthetic activations and eligibility regions; not a training-quality or trainer benchmark.",
        "capture_seconds": {},
        "policies": {},
        "samples_ms": {},
        "median_ms": {},
        "timing": {
            "windows": args.windows,
            "discard_first": 2,
            "replays_per_window": args.repeats,
            "order": "Seeded shuffle of all policies and all routes each window",
        },
    }
    results["software"] = {
        name: version(name)
        for name in ("torch", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl", "apache-tvm-ffi")
    }
    results["cuda"] = torch.version.cuda
    graphs, retained = {}, [values, do]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for policy in names:
            started = time.monotonic()
            mask, info = make_mask(policy, args.block, values)
            retained.append(mask)
            outputs = {}
            print("SETUP", args.block, policy, flush=True)
            for mode in modes:

                def fb():
                    o = cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode=mode)
                    return (o, *torch.autograd.grad(o, values, do))

                for _ in range(3):
                    fb()
                torch.cuda.synchronize()
                if args.profile:
                    o = cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode=mode)
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push(policy + "_" + mode + "_bwd")
                    _grads = torch.autograd.grad(o, values, do)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    print("PROFILE_DONE", policy, mode, flush=True)
                    return
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = fb()
                graphs[policy + "/" + mode + "_fb"] = graph
                outputs[mode] = out
                graph.replay()
                torch.cuda.synchronize()
                o = cudnn_flex_attention(*values, mask=mask, scale=128**-0.5, gqa_mode=mode)
                retained.append(o)

                def bw():
                    return torch.autograd.grad(o, values, do, retain_graph=True)

                for _ in range(3):
                    bw()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    grad = bw()
                graphs[policy + "/" + mode + "_bwd"] = graph
                retained.append(grad)
                graph.replay()
                torch.cuda.synchronize()
            errors = {}
            for mode in MODES:
                assert all(bool(torch.isfinite(x).all()) for x in outputs[mode])
                if mode == "native":
                    continue
                errors[mode] = {
                    name: float(
                        (
                            (a.detach().float() - b.detach().float()).norm()
                            / b.detach().float().norm()
                        ).item()
                    )
                    for name, a, b in zip(("o", "dq", "dk", "dv"), outputs[mode], outputs["native"])
                }
                # The candidate retains native forward exactly. External folding
                # also changes the forward layout, so record its diagnostic
                # differences separately instead of gating the candidate on them.
                if mode == "internal":
                    assert torch.equal(outputs[mode][0], outputs["native"][0]), (
                        policy,
                        mode,
                        errors,
                    )
                    assert max(errors[mode].values()) < 5e-4, (policy, mode, errors)
            retained.append(outputs)
            info["relative_l2_vs_native"] = errors
            info["native_5e4_comparison_passed"] = {
                mode: max(error.values()) < 5e-4 for mode, error in errors.items()
            }
            info["bitwise_output_vs_native"] = {
                mode: torch.equal(outputs[mode][0], outputs["native"][0])
                for mode in ("external", "internal")
            }
            info["tile_counts"] = tile_counts(mask, values, info["valid_score_pairs_all_heads"])
            results["policies"][policy] = info
            results["capture_seconds"][policy] = time.monotonic() - started
            print(
                "READY",
                args.block,
                policy,
                "GiB",
                torch.cuda.memory_allocated() / 2**30,
                "errors",
                errors,
                flush=True,
            )
    keys = list(graphs)
    times = {key: [] for key in keys}
    order_rng = random.Random(77191)
    for window in range(args.windows):
        order = keys.copy()
        order_rng.shuffle(order)
        for key in order:
            graph = graphs[key]
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                graph.replay()
            end.record()
            end.synchronize()
            if window >= 2:
                times[key].append(start.elapsed_time(end) / args.repeats)
        print("WINDOW", args.block, window, flush=True)
    results["samples_ms"] = times
    results["median_ms"] = {key: statistics.median(samples) for key, samples in times.items()}
    results["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    results["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print("DONE", args.block, json.dumps(results["median_ms"]), flush=True)


if __name__ == "__main__":
    main()
