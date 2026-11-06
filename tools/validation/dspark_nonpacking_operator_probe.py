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

"""Real-activation Q/K and cuDNN local backward against FP32 math, BF16 dO fixed."""

import argparse
import json
from pathlib import Path

from tools.validation.dspark_nonpacking_alignment import vector_comparison
from tools.validation.dspark_trajectory_probe import (
    build_model,
    load_state,
    load_target_head,
    prepare_rows,
)


def norm_rope(value, weight, positions, cos, sin, heads, dim, eps):
    """Independent RMSNorm/indexed-RoPE math; no intermediate BF16 rounding."""
    import torch

    value = value.reshape(*value.shape[:2], heads, dim)
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    value = value * weight
    first, second = value.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    cos, sin = cos.squeeze(0).squeeze(0), sin.squeeze(0).squeeze(0)
    return value * cos[positions].unsqueeze(2) + rotated * sin[positions].unsqueeze(2)


def interval_mask(endpoints, key_length):
    import torch

    if endpoints.ndim != 2 or endpoints.shape[0] != 3:
        raise ValueError("This oracle requires the minimal nonpacked three-endpoint mask")
    keys = torch.arange(key_length, device=endpoints.device).unsqueeze(0)
    return (keys < endpoints[0].unsqueeze(1)) | (
        (keys >= endpoints[1].unsqueeze(1)) & (keys < endpoints[2].unsqueeze(1))
    )


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError("Use a fresh output")
    torch.set_num_threads(8)
    from aspec.models.ops import cudnn_flex_attention as attention_api
    from aspec.models.ops import dspark_qk_norm_rope as qk_api

    fixture = Path(args.fixture)
    config = json.loads((fixture / "manifest.json").read_text())["draft_config"]
    head = load_target_head(fixture)
    rows, identities = prepare_rows(
        next((fixture / "cache/eval_cache").glob("*/eval_rank_0.pt")), 1, head
    )
    model = build_model(
        "aspec", "cudnn_flex_attention", config, load_state(args.checkpoint), "total"
    )
    captured = {}
    native_qk, native_attention = qk_api.dspark_qk_norm_rope, attention_api.cudnn_flex_attention

    def capture_qk(*values):
        result = native_qk(*values)
        if "qk" not in captured:
            record = {
                "values": tuple(
                    v.detach().clone() if isinstance(v, torch.Tensor) else v for v in values
                ),
                "outputs": [v.detach().clone() for v in result],
            }
            captured["qk"] = record
            for index, value in enumerate(result):
                value.register_hook(
                    lambda gradient, i=index: record.__setitem__(
                        "do" + str(i), gradient.detach().clone()
                    )
                )
        return result

    def capture_attention(*values, **options):
        result = native_attention(*values, **options)
        if "attention" not in captured:
            record = {
                "values": [v.detach().clone() for v in values],
                "options": dict(options),
                "output": result.detach().clone(),
            }
            captured["attention"] = record
            result.register_hook(
                lambda gradient: record.__setitem__("do", gradient.detach().clone())
            )
        return result

    qk_api.dspark_qk_norm_rope, attention_api.cudnn_flex_attention = capture_qk, capture_attention
    try:
        native_output = model(**rows[0], lm_head_weight=head.lm_head.weight)
        native_output[0].backward()
    finally:
        qk_api.dspark_qk_norm_rope, attention_api.cudnn_flex_attention = native_qk, native_attention
    del native_output, model, head, rows
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "documents": identities,
        "no_updates": True,
        "dO_dtype": "torch.bfloat16",
        "qk": {},
        "attention": {},
    }

    record = captured["qk"]
    q, ck, dk, qw, kw, eps, cos, sin, qp, cp, dp, hq, hk, dim = record["values"]
    actual_inputs = [value.detach().clone().requires_grad_() for value in (q, ck, dk, qw, kw)]
    actual_out = native_qk(*actual_inputs, eps, cos, sin, qp, cp, dp, hq, hk, dim)
    dos = (record["do0"], record["do1"])
    if any(value.dtype != torch.bfloat16 for value in dos):
        raise ValueError("Real Q/K output gradients were not BF16")
    actual_grads = torch.autograd.grad(actual_out, actual_inputs, grad_outputs=dos)
    oracle_inputs = [value.detach().float().requires_grad_() for value in (q, ck, dk, qw, kw)]
    oq, ock, odk, oqw, okw = oracle_inputs
    oracle_out = (
        norm_rope(oq, oqw, qp, cos.float(), sin.float(), hq, dim, eps),
        torch.cat(
            (
                norm_rope(ock, okw, cp, cos.float(), sin.float(), hk, dim, eps),
                norm_rope(odk, okw, dp, cos.float(), sin.float(), hk, dim, eps),
            ),
            dim=1,
        ),
    )
    oracle_grads = torch.autograd.grad(
        oracle_out, oracle_inputs, grad_outputs=[v.float() for v in dos]
    )
    result["qk"]["forward"] = vector_comparison(actual_out, oracle_out)
    result["qk"]["gradients"] = {
        name: vector_comparison([actual], [oracle])
        for name, actual, oracle in zip(
            ("q", "context_k", "draft_k", "q_weight", "k_weight"),
            actual_grads,
            oracle_grads,
            strict=True,
        )
    }

    record = captured["attention"]
    actual_inputs = [value.detach().clone().requires_grad_() for value in record["values"]]
    actual_out = native_attention(*actual_inputs, **record["options"])
    do = record["do"]
    if do.dtype != torch.bfloat16:
        raise ValueError("Real attention dO was not BF16")
    actual_grads = torch.autograd.grad(actual_out, actual_inputs, grad_outputs=do)
    oracle_inputs = [value.detach().float().requires_grad_() for value in record["values"]]
    oq, ok, ov = [value.transpose(1, 2) for value in oracle_inputs]
    ratio = oq.shape[1] // ok.shape[1]
    endpoints = record["options"]["mask"].mask_func[0]
    mask = interval_mask(endpoints, ok.shape[2])
    with sdpa_kernel(SDPBackend.MATH):
        oracle_out = torch.nn.functional.scaled_dot_product_attention(
            oq,
            ok.repeat_interleave(ratio, 1),
            ov.repeat_interleave(ratio, 1),
            attn_mask=mask,
            dropout_p=0.0,
            scale=record["options"]["scale"],
        ).transpose(1, 2)
    oracle_grads = torch.autograd.grad(oracle_out, oracle_inputs, grad_outputs=do.float())
    result["attention"]["forward"] = vector_comparison([actual_out], [oracle_out])
    result["attention"]["gradients"] = {
        name: vector_comparison([actual], [oracle])
        for name, actual, oracle in zip(("q", "k", "v"), actual_grads, oracle_grads, strict=True)
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
