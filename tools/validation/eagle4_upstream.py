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

"""Fixed-state EAGLE4 backbone/TTT gradients versus installed TorchSpec, not a benchmark."""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aspec.config.eagle4 import Eagle4Config
from aspec.models.draft.eagle4 import Eagle4DraftModel
from aspec.training.optimizer import BF16Optimizer
from tools.validation.eagle_thd_reference import difference


def objective(model, ids, features, targets):
    hidden = model.project_hidden_states(features)
    length = hidden.shape[1]
    positions = torch.arange(length, device=hidden.device)[None]
    mask = model.prepare_decoder_attention_mask(torch.ones_like(ids[:, :length]), hidden, 1, length, 0)
    keys = values = None
    losses, outputs = [], []
    for depth in range(3):
        hidden, keys, values = model.backbone(
            model.embed_input_ids(ids[:, depth + 1:depth + length + 1]),
            hidden, mask, positions, keys, values, True)
        outputs.append(hidden)
        logits = model.compute_logits(hidden).float()
        valid = (torch.arange(length, device=hidden.device) + depth + 1 < length)
        losses.append(F.kl_div(logits.log_softmax(-1), targets, reduction="none").sum(-1)[:, valid].mean())
        hidden = model.next_depth_hidden(hidden)
    return losses, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=16, type=int)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3

    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    records = []
    for variant in ("dense", "sparse_swiglu", "nemotron_latent_relu2"):
        torch.manual_seed(341)
        raw = dict(architectures=["LlamaForCausalLMEagle4"], hidden_size=32,
                   intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=128, vocab_size=97, draft_vocab_size=97,
                   num_aux_hidden_states=2, target_hidden_size=32, pad_token_id=0,
                   rms_norm_eps=1e-6, max_position_embeddings=128, pretraining_tp=1,
                   use_rope=False, position_embedding_type="none", fc_norm=True,
                   norm_output=variant != "nemotron_latent_relu2",
                   attn_norm=variant == "nemotron_latent_relu2",
                   enable_post_mlp_norm=variant == "nemotron_latent_relu2",
                   sliding_window=4, use_sliding_window=True, layer_types=["sliding_attention"],
                   efficient_lm_head_config={"type": "low_rank", "rank": 8, "num_candidates": 16})
        if variant != "dense":
            raw.update(enable_moe=True, moe_architecture=variant, n_routed_experts=4,
                       num_experts_per_tok=2, moe_intermediate_size=64, n_shared_experts=1,
                       moe_latent_size=32, moe_shared_expert_intermediate_size=64,
                       hidden_act="relu2" if variant == "nemotron_latent_relu2" else "silu")
        cfg = Eagle4Config(**raw)
        actual = Eagle4DraftModel(cfg, attention_backend="sdpa").cuda().bfloat16()
        actual.configure_training_stage()
        reference = LlamaForCausalLMEagle3(cfg, attention_backend="sdpa").cuda().bfloat16()
        reference.freeze_embedding()
        settings = dict(lr=6e-4, total_steps=32, warmup_ratio=0.0, weight_decay=0.0)
        optimizer, ref_optimizer = BF16Optimizer(actual, **settings), BF16Optimizer(reference, **settings)
        names = [n for n, p in actual.named_parameters() if p.requires_grad]
        for step in range(args.steps):
            reference.load_state_dict({k: v for k, v in actual.state_dict().items()
                                       if k in reference.state_dict()}, strict=True)
            ref_optimizer.sync_fp32_params_from_model()
            for a, b in zip(ref_optimizer.fp32_params, optimizer.fp32_params, strict=True):
                a.data.copy_(b.data)
            ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            ref_optimizer.scheduler.load_state_dict(optimizer.scheduler.state_dict())
            torch.manual_seed(104 + step)
            ids = torch.randint(1, 96, (1, 14), device="cuda")
            features = torch.randn(1, 11, 64, device="cuda", dtype=torch.bfloat16)
            targets = torch.randn(1, 11, 97, device="cuda").softmax(-1)
            losses, output = objective(actual, ids, features, targets)
            ref_losses, ref_output = objective(reference, ids, features, targets)
            sum(losses).backward()
            sum(ref_losses).backward()
            a, b = dict(actual.named_parameters()), dict(reference.named_parameters())
            record = {"variant": variant, "step": step + 1, "loss": difference(losses, ref_losses),
                      "hidden": difference(output, ref_output),
                      "gradients": difference([a[n].grad for n in names], [b[n].grad for n in names])}
            optimizer.step()
            ref_optimizer.step()
            record["updated_parameters"] = difference([a[n] for n in names], [b[n] for n in names])
            if record["loss"]["max_absolute"] > 1e-5 or record["gradients"]["relative_l2"] > 0.01:
                raise AssertionError(record)
            records.append(record)
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
        print({k: records[-1][k] for k in ("variant", "step", "loss", "gradients")}, flush=True)
    dist.destroy_process_group()
    args.output.write_text(json.dumps({"synthetic_data": True, "reference": "TorchSpec a4e5c10",
                                      "control_bias_update": "covered separately", "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
