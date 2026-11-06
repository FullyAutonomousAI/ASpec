# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture real DSpark attention inputs, then replay layout alternatives.

Capture accepts the native ``torchspec_benchmark run`` arguments. Replay uses
the same BF16 inputs and upstream gradient for timing and numerical comparisons.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def capture(args, remaining):
    import torch

    from aspec.models.ops import cudnn_flex_attention as attention
    from aspec.models.ops import dspark_qk_norm_rope as qk
    from aspec.training.dspark_trainer import DSparkTrainer
    from tools.performance import torchspec_benchmark

    original_step = DSparkTrainer.train_from_queue
    original_attention = attention.cudnn_flex_attention
    original_qk = qk.dspark_qk_norm_rope
    active = False
    snapshot = {}

    def observed_attention(query, key, value, **kwargs):
        output = original_attention(query, key, value, **kwargs)
        if active and "attention" not in snapshot:
            sample = {
                "q": query.detach().clone(), "k": key.detach().clone(),
                "v": value.detach().clone(),
                "mask": kwargs["mask"].mask_func.detach().clone(),
                "scale": kwargs["scale"],
            }
            snapshot["attention"] = sample
            output.register_hook(lambda grad: sample.update(dout=grad.detach().clone()))
        return output

    def observed_qk(*positional, **kwargs):
        output = original_qk(*positional, **kwargs)
        if active and "qk" not in snapshot:
            sample = {
                "args": tuple(x.detach().clone() if isinstance(x, torch.Tensor) else x
                              for x in positional)
            }
            snapshot["qk"] = sample
            output[0].register_hook(lambda grad: sample.update(dq=grad.detach().clone()))
            output[1].register_hook(lambda grad: sample.update(dk=grad.detach().clone()))
        return output

    def observed_step(self, *positional, **kwargs):
        nonlocal active
        active = kwargs.get("step") == args.capture_step
        result = original_step(self, *positional, **kwargs)
        active = False
        if snapshot and not args.snapshot.exists():
            def cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, tuple):
                    return tuple(cpu(v) for v in value)
                return value
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cpu(snapshot), args.snapshot)
            print("Captured real attention/QK inputs: " + str(args.snapshot), flush=True)
        return result

    if args.snapshot.exists():
        raise FileExistsError(args.snapshot)
    attention.cudnn_flex_attention = observed_attention
    qk.dspark_qk_norm_rope = observed_qk
    DSparkTrainer.train_from_queue = observed_step
    sys.argv = [sys.argv[0], *remaining]
    try:
        torchspec_benchmark.main()
        if not args.snapshot.exists():
            raise RuntimeError("Requested capture step was not executed")
    finally:
        attention.cudnn_flex_attention = original_attention
        qk.dspark_qk_norm_rope = original_qk
        DSparkTrainer.train_from_queue = original_step


def tensor_metrics(reference, value):
    import torch

    ref, got = reference.float(), value.float()
    delta = got - ref
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rel_l2": (delta.norm() / ref.norm().clamp_min(1e-30)).item(),
        "all_finite": bool(torch.isfinite(got).all()),
    }


def replay(args):
    import torch

    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    snapshot = torch.load(args.snapshot, map_location="cuda", weights_only=True)
    sample = snapshot["attention"]
    q, k, v = (sample[key].detach() for key in ("q", "k", "v"))
    mask = CudnnFlexMask(sample["mask"])
    candidates = {}
    reference = None
    for layout in args.layouts:
        options = {
            "physical": {"physical_gqa": True, "pack_gqa": False},
            "native-packed": {"physical_gqa": False, "pack_gqa": True},
            "native-unpacked": {"physical_gqa": False, "pack_gqa": False},
        }[layout]

        def run():
            # Fresh leaves bind AccumulateGrad to this invocation's stream,
            # including CUDA graph capture on its side stream.
            inputs = tuple(t.detach().requires_grad_() for t in (q, k, v))
            output = cudnn_flex_attention(*inputs, mask=mask, scale=sample["scale"], **options)
            gradients = torch.autograd.grad(output, inputs, sample["dout"])
            return output, *gradients

        result = run()
        torch.cuda.synchronize()
        if reference is None:
            reference = tuple(t.detach().clone() for t in result)
        errors = {name: tensor_metrics(ref, got)
                  for name, ref, got in zip(("output", "dq", "dk", "dv"), reference, result)}
        del result
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_result = run()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        measurements = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            measurements.append(start.elapsed_time(end) / args.iterations)
        candidates[layout] = {"milliseconds": measurements,
                              "median_ms": statistics.median(measurements), "errors": errors}
        print(json.dumps({layout: candidates[layout]}), flush=True)
        del graph, static_result
    payload = {"device": torch.cuda.get_device_name(),
               "capability": torch.cuda.get_device_capability(),
               "qkv_shapes": [list(t.shape) for t in (q, k, v)],
               "snapshot": str(args.snapshot), "candidates": candidates,
               "scope": "Captured native BF16 attention forward/backward; CUDA graph device timing"}
    if args.result.exists():
        raise FileExistsError(args.result)
    args.result.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--capture-step", type=int, default=8)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--layouts", nargs="+", default=("physical", "native-packed", "native-unpacked"),
                        choices=("physical", "native-packed", "native-unpacked"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args, remaining = parser.parse_known_args()
    if args.mode == "capture":
        # The result argument belongs to the existing benchmark parser in this mode.
        if args.result:
            remaining += ["--result", str(args.result)]
        capture(args, remaining)
    else:
        if remaining or args.result is None:
            parser.error("replay requires --result and accepts no benchmark arguments")
        replay(args)


if __name__ == "__main__":
    main()
