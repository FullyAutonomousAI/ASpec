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

"""Native MTP against fixed, unmodified Speculators mathematics (no benchmark)."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch

from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel
from aspec.models.mtp import MTPModel, mtp_step_weights
from aspec.training.optimizer import BF16Optimizer
from tools.validation.eagle_thd_reference import difference

REVISION = "7ec7aba9450b533334e8f7992baf4e93be3c1a57"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--common-states", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(["git", "-C", str(args.reference_root), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != REVISION:
        raise ValueError("Speculators reference revision mismatch")
    sys.path[:0] = [str(args.reference_root / "src"),
                   str(args.reference_root / "hs_connectors/src")]
    # Import an uninstalled, fixed source checkout without installing it into
    # the project's shared venv. Only package-version bookkeeping is supplied;
    # no reference math, decoder, weights or training function is patched.
    original_version = importlib.metadata.version
    importlib.metadata.version = lambda name: "0.0.0" if name == "speculators" else original_version(name)
    try:
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.convert.mtp.converter import MTPConverter
        from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
        from speculators.proposals.greedy import GreedyTokenProposalConfig
    finally:
        importlib.metadata.version = original_version
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    native = {
        "model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
        "vocab_size": 97, "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "rms_norm_eps": 1e-6, "attention_dropout": 0.0, "attention_bias": False,
        "hidden_act": "silu", "max_position_embeddings": 128, "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                            "partial_rotary_factor": 0.25, "mrope_section": [4, 6, 6],
                            "mrope_interleaved": True},
    }
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(411)
        draft = NativeMTPDraftModel(NativeMTPConfig(native_config=native)).cuda().to(dtype)
        actual = MTPModel(draft, length=3, gradient_checkpointing=True)
        config = MTPSpeculatorConfig(
            transformer_layer_config=Qwen3_5TextConfig(**native),
            speculators_config=SpeculatorsConfig(
                algorithm="mtp", proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(name_or_path="synthetic", architectures=["Qwen3_5ForConditionalGeneration"]),
            ),
        )
        reference = MTPDraftModel(config).cuda().to(dtype)
        def reference_key(key):
            if key == "embed_tokens.weight":
                return key
            if key == "lm_head.weight":
                return key
            return MTPConverter._remap_key(draft.native_key(key))
        mapped = {reference_key(key): tensor for key, tensor in draft.state_dict().items()}
        missing, unexpected = reference.load_state_dict(mapped, strict=False)
        if unexpected or any(key not in {"verifier_lm_head.weight", "d2t", "t2d"} for key in missing):
            raise AssertionError((missing, unexpected))
        actual_params = {name: param for name, param in draft.named_parameters() if param.requires_grad}
        ref_params = dict(reference.named_parameters())
        assert all(not ref_params[key].requires_grad for key in ("embed_tokens.weight", "lm_head.weight"))
        settings = dict(lr=1e-4, total_steps=32, warmup_ratio=0.0, weight_decay=0.0)
        optimizer, ref_optimizer = BF16Optimizer(draft, **settings), BF16Optimizer(reference, **settings)
        a_masters = dict(zip(actual_params, optimizer.fp32_params, strict=True))
        b_masters = dict(zip((n for n, p in reference.named_parameters() if p.requires_grad),
                             ref_optimizer.fp32_params, strict=True))
        weights = mtp_step_weights(3, 0.6)
        for step in range(args.steps):
            if args.common_states:
                reference.load_state_dict({reference_key(k): v for k, v in draft.state_dict().items()},
                                          strict=False)
                for name, master in a_masters.items():
                    b_masters[reference_key(name)].data.copy_(master)
                    # State is keyed by the corresponding native parameter,
                    # not by the two models' potentially different traversal order.
                    state = optimizer.state.get(master, {})
                    ref_optimizer.state[b_masters[reference_key(name)]] = {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for key, value in state.items()
                    }
            torch.manual_seed(501 + step)
            ids = torch.randint(0, 97, (1, 29), device="cuda")
            hidden = torch.randn(1, 29, 32, device="cuda", dtype=dtype)
            mask = torch.ones_like(ids)
            mask[:, :5] = 0
            result = actual(ids, hidden, mask, torch.ones_like(ids))
            loss = sum(w * value for w, value in zip(weights, result[0], strict=True))
            _, ref_loss, metrics = reference(input_ids=ids, hidden_states=hidden,
                                             loss_mask=mask, step_weights=weights)
            loss.backward()
            ref_loss.backward()
            per_tensor = {name: difference([p.grad], [ref_params[reference_key(name)].grad])
                          for name, p in actual_params.items()}
            row = {
                "dtype": str(dtype), "step": step + 1,
                "loss": difference([loss], [ref_loss]),
                "per_depth_loss": difference(result[1], [metrics[f"loss_step_{k}"] / w
                                                          for k, w in enumerate(weights)]),
                "gradients": difference([p.grad for p in actual_params.values()],
                                        [ref_params[reference_key(n)].grad for n in actual_params]),
                "per_tensor_gradients": per_tensor,
            }
            optimizer.step()
            ref_optimizer.step()
            row["parameters"] = difference(list(actual_params.values()),
                                           [ref_params[reference_key(n)] for n in actual_params])
            row["master"] = difference(list(a_masters.values()),
                                      [b_masters[reference_key(n)] for n in actual_params])
            row["moments"] = {key: difference(
                [optimizer.state[a_masters[n]][key] for n in actual_params],
                [ref_optimizer.state[b_masters[reference_key(n)]][key] for n in actual_params],
            ) for key in ("exp_avg", "exp_avg_sq")}
            records.append(row)
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            print({k: v for k, v in row.items() if k not in {"per_tensor_gradients", "moments"}}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference_revision": REVISION, "synthetic_data": True,
                                      "steps": args.steps, "common_states": args.common_states,
                                      "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
