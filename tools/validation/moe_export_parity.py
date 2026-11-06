#!/usr/bin/env python3
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

"""Compare an exported ASpec MoE layer with the patched SGLang fused loader.

This is a single-GPU artifact check, not a throughput benchmark.  It builds the
official SGLang MoE implementation selected by ``ffn_type``, loads one layer
through the same fused-expert parameter mapping as the production DFlash
loader, and evaluates it on the exact hidden states used by the ASpec
training module.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_sglang(export_dir: Path) -> None:
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path=str(export_dir),
        skip_tokenizer_init=True,
        tp_size=1,
        attention_backend="torch_native",
    )
    publish(args, role="scheduler")
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
        backend="nccl",
    )
    get_parallel().override_permanently(
        tp_rank=0,
        pp_rank=0,
        attn_dp_rank=0,
        attn_tp_rank=0,
        attn_cp_rank=0,
        moe_ep_rank=0,
        moe_dp_rank=0,
        moe_tp_rank=0,
        attn_dcp_rank=0,
    )
    initialize_model_parallel()


def _layer_state(path: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer}.mlp."
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {
            name.removeprefix(prefix): handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith(prefix)
        }
    if not state:
        raise ValueError(f"No MoE tensors found for layer {layer} in {path}")
    return state


def _load_sglang_moe(module: torch.nn.Module, state: dict[str, torch.Tensor], experts: int) -> None:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    expert_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=experts,
    )
    stacked_mapping = (
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
    params = dict(module.named_parameters())
    loaded_source: set[str] = set()

    for name, weight in state.items():
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if weight_name not in name:
                continue
            resolved = name.replace(weight_name, param_name)
            if resolved not in params:
                raise ValueError(f"Expert parameter {name!r} mapped to absent {resolved!r}")
            param = params[resolved]
            param.weight_loader(
                param,
                weight,
                resolved,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            loaded_source.add(name)
            break
        else:
            for param_name, weight_name, shard_id in stacked_mapping:
                if ".experts." in name or f".{weight_name}." not in f".{name}":
                    continue
                resolved = name.replace(weight_name, param_name)
                if resolved not in params:
                    continue
                param = params[resolved]
                getattr(param, "weight_loader", default_weight_loader)(param, weight, shard_id)
                loaded_source.add(name)
                break
            else:
                if name not in params:
                    raise ValueError(f"Non-expert parameter {name!r} has no SGLang destination")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded_source.add(name)

    missing = set(state) - loaded_source
    if missing:
        raise ValueError(f"SGLang parity loader skipped source tensors: {sorted(missing)}")


def _reference_moe(config, ffn_type: str, layer: int) -> torch.nn.Module:
    from aspec.models.draft.ling_moe import LingDraftMoE
    from aspec.models.draft.moe import QwenDraftMoE

    if ffn_type == "qwen3_moe":
        return QwenDraftMoE(config)
    if ffn_type == "ling_moe":
        return LingDraftMoE(config, layer_idx=layer)
    raise ValueError(f"Expected qwen3_moe or ling_moe, got {ffn_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=37)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    raw = json.loads((export_dir / "config.json").read_text())
    ffn_type = raw.get("ffn_type", "dense")
    if ffn_type not in {"qwen3_moe", "ling_moe"}:
        raise ValueError(f"Export is not a supported MoE artifact: ffn_type={ffn_type!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires one CUDA GPU")

    _initialize_sglang(export_dir)
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.dflash import _build_dflash_mlp

    from aspec.models.draft.dflash import DFlashConfig

    config = DFlashConfig(**raw)
    state = _layer_state(export_dir / "model.safetensors", args.layer)
    reference = _reference_moe(config, ffn_type, args.layer)
    reference.load_state_dict(state, strict=True)
    reference = reference.to(device="cuda", dtype=torch.bfloat16).eval()

    serving_config = AutoConfig.from_pretrained(export_dir, trust_remote_code=False)
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        serving = _build_dflash_mlp(
            serving_config,
            layer_id=args.layer,
            prefix="",
        )
    _load_sglang_moe(serving, state, int(serving_config.num_experts))
    serving.eval()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    hidden = torch.randn(
        args.tokens,
        int(serving_config.hidden_size),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        expected, observation = reference(hidden)
        # Some fused MoE runners reuse their input buffer. Keep the canonical
        # activation intact for the independent router comparison below.
        actual = serving(hidden.clone(), None)
        if ffn_type == "qwen3_moe":
            reference_logits = observation
            reference_ids = reference_logits.float().topk(
                int(serving_config.num_experts_per_tok), dim=-1
            ).indices
            serving_logits, _ = serving.gate(hidden)
        else:
            reference_logits, reference_ids = observation
            serving_logits = serving.gate(hidden)
        serving_topk = serving.topk(hidden, serving_logits)
        if not hasattr(serving_topk, "topk_ids"):
            raise RuntimeError(
                "The selected SGLang routing backend did not expose standard top-k ids"
            )
        serving_ids = serving_topk.topk_ids
    difference = actual.float() - expected.float()
    expected_norm = torch.linalg.vector_norm(expected.float())
    logit_difference = serving_logits.float() - reference_logits.float()
    gate_weight_difference = (
        serving.gate.weight.float() - reference.gate.weight.float()
    )
    routes_equal = bool(
        torch.equal(
            serving_ids.long().sort(dim=-1).values,
            reference_ids.long().sort(dim=-1).values,
        )
    )
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / expected_norm.clamp_min(1e-12)
    )
    max_abs = float(difference.abs().max())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    passed = bool(
        finite
        and routes_equal
        and relative_l2 <= args.max_relative_l2
        and max_abs <= args.max_abs
    )
    report = {
        "schema_version": "aspec.moe-export-parity/v1",
        "export_dir": str(export_dir),
        "ffn_type": ffn_type,
        "layer": args.layer,
        "tokens": args.tokens,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(hidden.dtype),
        "source_tensor_count": len(state),
        "finite": finite,
        "routing_ids_equal": routes_equal,
        "reference_gate_dtype": str(reference.gate.weight.dtype),
        "serving_gate_dtype": str(serving.gate.weight.dtype),
        "gate_weight_max_abs": float(gate_weight_difference.abs().max().detach()),
        "router_logits_max_abs": float(logit_difference.abs().max()),
        "max_abs": max_abs,
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": relative_l2,
        "thresholds": {
            "max_abs": args.max_abs,
            "max_relative_l2": args.max_relative_l2,
        },
        "passed": passed,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not passed:
        raise SystemExit("MoE export parity check failed; see the emitted report")


if __name__ == "__main__":
    # Prevent a caller's world settings from silently turning this TP1 check
    # into a different distributed topology.
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
