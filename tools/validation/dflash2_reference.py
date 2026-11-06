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

"""Compare DFlash2 backbone/gradients and selector with unmodified Z Lab code.

The reference treats anchors as separate batch items, so its native convolution
cannot cross blocks. It receives an explicit training-time prefix/window mask.
No unpublished training objective is inferred from this comparison.
"""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import torch
from transformers import Qwen3Config

from aspec.models.dflash2 import DFlash2Model
from aspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel

REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"


def difference(actual, expected):
    a = torch.cat([v.detach().float().reshape(-1) for v in actual])
    b = torch.cat([v.detach().float().reshape(-1) for v in expected])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite reference result")
    return {"max_absolute": (a - b).abs().max().item(),
            "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("flex_attention", "fa4"), default="flex_attention")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected fixed Z Lab revision {REVISION}, got {revision}")
    source = args.reference_root / "dflash/model.py"
    spec = importlib.util.spec_from_file_location("zlab_dflash2_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for dtype in ((torch.bfloat16,) if args.backend == "fa4" else (torch.float32, torch.bfloat16)):
        for window, causal in ((None, False), (5, False), (5, True)):
            torch.manual_seed(211)
            batch, length, anchors, block, hidden, vocab = 2, 40, 3, 8, 32, 97
            config = DFlash2Config(
                hidden_size=hidden, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                vocab_size=vocab, mask_token_id=96, block_size=block,
                num_target_layers=2, target_num_hidden_layers=8,
                target_layer_ids=[1, 5], target_hidden_size=hidden,
                max_position_embeddings=128, conv_group_size=4, selector_rank=8,
                selector_top_k=4, rope_theta=1000000.0, is_causal=causal,
                layer_types=["sliding_attention" if window else "full_attention"] * 2,
                sliding_window=window,
            )
            draft = DFlash2DraftModel(config).cuda().to(dtype)
            draft.freeze_embedding()
            with torch.no_grad():
                for layer in draft.layers:
                    for conv in (layer.attention_conv, layer.mlp_conv):
                        conv.kernel_projection.weight.normal_(std=0.02)
                        conv.base_kernel.add_(torch.randn_like(conv.base_kernel) * 0.03)
            raw = draft.config_for_serving()
            expected = reference.DFlash2DraftModel(Qwen3Config(**raw)).cuda().to(dtype)
            weights = draft.state_dict_for_serving()
            weights = {key + ".weight" if key.endswith(("predecessor_codebook", "successor_codebook")) else key: value
                       for key, value in weights.items()}
            expected.load_state_dict(weights, strict=True)
            actual = DFlash2Model(draft, block_size=block, num_anchors=anchors, attention_backend=args.backend)
            positions = torch.tensor([[6, 17, 29]] * batch, device="cuda")
            keep = torch.ones_like(positions, dtype=torch.bool)
            ids = torch.randint(0, vocab - 1, (batch, length), device="cuda")
            features = torch.randn(batch, length, hidden * 2, device="cuda", dtype=dtype,
                                   requires_grad=True)
            ref_features = features.detach().clone().requires_grad_(True)
            output = actual._draft_backbone(
                ids, features, torch.ones_like(ids), injected_anchors=positions,
                injected_keep_mask=keep,
            )[0]
            noise = actual._create_noise_embed(ids, positions, keep).detach()
            offsets = torch.arange(block, device="cuda")
            query_positions = (positions[..., None] + offsets).reshape(batch * anchors, block)
            context_positions = torch.arange(length, device="cuda").expand(batch * anchors, -1)
            context_visible = context_positions[:, None, :] < positions.reshape(-1, 1, 1)
            context_visible = context_visible.expand(-1, block, -1)
            draft_visible = torch.ones(batch * anchors, block, block, device="cuda", dtype=torch.bool)
            if window:
                context_visible = context_visible & (query_positions[:, :, None] - context_positions[:, None, :] < window)
                draft_visible &= (offsets[:, None] - offsets[None, :]).abs() < window
            if causal:
                draft_visible &= offsets[:, None] >= offsets[None, :]
            mask = torch.cat((context_visible, draft_visible), dim=-1).unsqueeze(1)
            reference_output = expected(
                target_hidden=ref_features[:, None].expand(-1, anchors, -1, -1).reshape(batch * anchors, length, -1),
                noise_embedding=noise.reshape(batch * anchors, block, hidden),
                position_ids=torch.cat((context_positions, query_positions), dim=1),
                attention_mask=mask,
            ).reshape_as(output)
            probe = torch.randn_like(output)
            (output * probe).float().sum().backward()
            (reference_output * probe).float().sum().backward()
            expected_params = dict(expected.named_parameters())
            gradients, ref_gradients = [features.grad], [ref_features.grad]
            per_tensor = {"target_features": difference([features.grad], [ref_features.grad])}
            for name, parameter in draft.named_parameters():
                if parameter.grad is None:
                    continue
                exported = next(iter(draft.state_dict_for_serving({name: parameter.grad})))
                gradients.append(parameter.grad)
                ref_gradients.append(expected_params[exported].grad)
                per_tensor[name] = difference([parameter.grad], [expected_params[exported].grad])
            # Compare the actual public selector's greedy self-conditioned path.
            logits = torch.randn(batch, block - 1, vocab, device="cuda", dtype=dtype)
            selector_hidden = torch.randn(batch, block - 1, hidden, device="cuda", dtype=dtype)
            predecessor = ids[:, 0]
            path = []
            for i in range(block - 1):
                scores, candidates = draft.candidate_selector.score_candidates(
                    selector_hidden[:, i], logits[:, i], predecessor,
                )
                predecessor = candidates.gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
                path.append(predecessor)
            expected_path = expected.candidate_selector.select(selector_hidden, logits, ids[:, 0], 0.0)[0]
            path_equal = torch.equal(torch.stack(path, dim=1), expected_path)
            result = {"dtype": str(dtype), "window": window, "causal": causal,
                      "output": difference([output], [reference_output]),
                      "gradients": difference(gradients, ref_gradients),
                      "per_tensor_gradients": per_tensor,
                      "gradient_tensors": len(gradients), "selector_path_equal": path_equal}
            results.append(result)
            tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-5
            if (not path_equal
                or any(result[key]["relative_l2"] > tolerance for key in ("output", "gradients"))
                or any(value["relative_l2"] > tolerance for value in per_tensor.values())):
                raise AssertionError(result)
            print({key: value for key, value in result.items() if key != "per_tensor_gradients"}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference_revision": revision, "backend": args.backend,
                                     "scope": "backbone, gradients, selector; not an official training objective",
                                     "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
