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

"""Native EAGLE FA4 THD/CP updates against document-isolated SDPA TTT."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import LlamaConfig

from aspec.data.utils import DFlashPackingCollator
from aspec.inference.context import context_shard_bounds
from aspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from aspec.models.eagle3 import (
    Eagle3Model,
    LazyTarget,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from aspec.models.ops.eagle_thd import EagleTHDLayout
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.tensor import padding


def difference(actual, reference):
    a = torch.cat([v.detach().float().flatten() for v in actual])
    b = torch.cat([v.detach().float().flatten() for v in reference])
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite result")
    return {"max_absolute": (a - b).abs().max().item(),
            "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item()}


def gate_probe(actual, reference, teacher, draft, kind, eta):
    """Separate LK kink sensitivity from smooth loss-head arithmetic (CP1)."""
    from aspec.models.ops.loss import _rmsnorm_logits, _softmax_from_logits

    norm, head, eps = draft.get_lm_head_params()
    up = draft.get_lm_head_up_weight()
    p = _softmax_from_logits(F.linear(teacher[0], teacher[1])).detach()
    a = _rmsnorm_logits(actual, norm, head, up, eps).detach().float().requires_grad_()
    b = _rmsnorm_logits(reference, norm, head, up, eps).detach().float().requires_grad_()
    qa, qb = _softmax_from_logits(a), _softmax_from_logits(b)
    below, equal = (qa.detach() < p), (qa.detach() == p)
    slope = below.float() + equal.float() * 0.5
    def objective(logits, q, fixed):
        alpha = (torch.minimum(p, q) if not fixed else
                 slope * q + (1 - slope) * p).sum(-1)
        if kind == "lk_alpha":
            return -alpha.clamp_min(1e-8).log().mean()
        lam = (-eta * alpha.detach().mean()).exp()
        kl = F.kl_div(F.log_softmax(logits, -1), p, reduction="none").sum(-1)
        tv = ((p - q).abs() if not fixed else
              (p - qa.detach()).sign() * (p - q)).sum(-1) * 0.5
        return (lam * kl + (1 - lam) * tv).mean()
    ga = torch.autograd.grad(objective(a, qa, False), a, retain_graph=True)[0]
    gb = torch.autograd.grad(objective(b, qb, False), b, retain_graph=True)[0]
    fixed = torch.autograd.grad(objective(b, qb, True), b)[0]
    return {
        "probabilities": difference([qa], [qb]),
        "branch_flip_fraction": ((qa.detach() < p) != (qb.detach() < p)).float().mean().item(),
        "loss_head_gradient": difference([ga], [gb]),
        "same_branch_gradient": difference([ga], [fixed]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mla", action="store_true")
    parser.add_argument("--eagle4-stage", choices=("draft", "candidate_head"))
    parser.add_argument("--eagle4-moe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--loss-type", default="all")
    parser.add_argument("--common-states", action="store_true")
    parser.add_argument("--gate-diagnostics", action="store_true")
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--short-documents", action="store_true")
    parser.add_argument("--draft-vocab-size", type=int, default=97)
    parser.add_argument("--reference-backend", choices=("sdpa", "fa4"), default="sdpa")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    if args.gate_diagnostics and (size != 1 or args.reference_backend != "sdpa"):
        raise ValueError("Gate diagnostic compares CP1 FA4 against SDPA")
    records = []
    kinds = (("forward_kl",) if args.eagle4_stage == "candidate_head" else
             ("forward_kl", "lk_alpha", "lk_lambda")) if args.loss_type == "all" else (args.loss_type,)
    for kind in kinds:
        torch.manual_seed(29)
        config = LlamaConfig(
            architectures=["LlamaForCausalLMEagle3"], hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=args.query_heads,
            num_key_value_heads=args.kv_heads, head_dim=128, vocab_size=97, draft_vocab_size=args.draft_vocab_size,
            target_hidden_size=32, num_aux_hidden_states=2, max_position_embeddings=128,
            pad_token_id=0, rms_norm_eps=1e-6, tie_word_embeddings=False,
        )
        draft_class = LlamaForCausalLMEagle3
        if args.mla:
            from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

            from aspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM

            raw = config.to_dict()
            raw.pop("model_type", None)
            raw.update(architectures=["Eagle3DeepseekV2ForCausalLM"],
                       num_key_value_heads=args.query_heads, q_lora_rank=16,
                       kv_lora_rank=16, qk_nope_head_dim=128, qk_rope_head_dim=64,
                       v_head_dim=128, pretraining_tp=1)
            config = DeepseekV3Config(**raw)
            draft_class = Eagle3DeepseekV2ForCausalLM
        wrapper_class = Eagle3Model
        if args.eagle4_stage:
            if args.mla or args.draft_vocab_size != config.vocab_size:
                raise ValueError("EAGLE4 uses its full-vocabulary MHA/GQA backbone")
            from aspec.config.eagle4 import Eagle4Config
            from aspec.models.draft.eagle4 import Eagle4DraftModel
            from aspec.models.eagle4 import Eagle4TrainingModel

            raw = config.to_dict()
            raw.update(architectures=["LlamaForCausalLMEagle4"],
                       efficient_lm_head_config={"type": "low_rank", "rank": 8, "num_candidates": 16},
                       candidate_teacher_top_k=8, eagle4_training_stage=args.eagle4_stage,
                       use_rope=False, fc_norm=True, norm_output=True,
                       sliding_window=4, use_sliding_window=True, layer_types=["sliding_attention"])
            if args.eagle4_moe:
                raw.update(enable_moe=True, n_routed_experts=4, num_experts_per_tok=2,
                           moe_intermediate_size=64, n_shared_experts=1)
            config, draft_class, wrapper_class = Eagle4Config(**raw), Eagle4DraftModel, Eagle4TrainingModel
        draft = draft_class(config, attention_backend="fa4").cuda().bfloat16()
        if args.eagle4_stage:
            draft.configure_training_stage()
        draft.freeze_embedding()
        if args.draft_vocab_size != config.vocab_size:
            selected = torch.linspace(0, config.vocab_size - 1, args.draft_vocab_size,
                                      device="cuda").long()
            t2d = torch.zeros(config.vocab_size, device="cuda", dtype=torch.bool)
            t2d[selected] = True
            draft.set_vocab_buffers(selected - torch.arange(args.draft_vocab_size, device="cuda"), t2d)
        else:
            t2d = None
        ref_draft = draft_class(config, attention_backend=args.reference_backend).cuda().bfloat16()
        ref_draft.freeze_embedding()
        if args.eagle4_stage:
            ref_draft.configure_training_stage()
        ref_draft.load_state_dict(draft.state_dict())
        model = wrapper_class(draft, length=3, attention_backend="fa4", loss_type=kind,
                              gradient_checkpointing=bool(args.eagle4_stage) and args.eagle4_stage != "candidate_head")
        reference = wrapper_class(ref_draft, length=3, attention_backend=args.reference_backend, loss_type=kind,
                                  gradient_checkpointing=bool(args.eagle4_stage) and args.eagle4_stage != "candidate_head")
        captured, ref_captured = [], []
        if args.gate_diagnostics:
            draft.midlayer.register_forward_hook(lambda _m, _i, out: captured.append(out[0].detach()))
            ref_draft.midlayer.register_forward_hook(lambda _m, _i, out: ref_captured.append(out[0].detach()))
        settings = dict(lr=6e-4, total_steps=32, warmup_ratio=0.0, weight_decay=0.0)
        optimizer, ref_optimizer = BF16Optimizer(draft, **settings), BF16Optimizer(ref_draft, **settings)
        for step in range(args.steps):
            if args.common_states:
                ref_draft.load_state_dict(draft.state_dict())
                ref_optimizer.sync_fp32_params_from_model()
                for destination, source in zip(ref_optimizer.fp32_params, optimizer.fp32_params, strict=True):
                    destination.data.copy_(source.data)
                ref_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                ref_optimizer.scheduler.load_state_dict(optimizer.scheduler.state_dict())
            captured.clear()
            ref_captured.clear()
            torch.manual_seed(103 + step)
            lengths = (3, 1) if args.short_documents else (17, 11)
            width = max(lengths)
            ids = torch.randint(1, 96, (2, width), device="cuda")
            auxiliary = torch.randn(2, width, 64, device="cuda", dtype=torch.bfloat16)
            teacher = torch.randn(2, width, 32, device="cuda", dtype=torch.bfloat16)
            head = torch.randn(97, 32, device="cuda", dtype=torch.bfloat16) * 0.1
            attention = torch.zeros(2, width, device="cuda", dtype=torch.long)
            mask = torch.zeros_like(attention)
            samples = []
            for doc, n in enumerate(lengths):
                attention[doc, :n] = 1
                mask[doc, (0 if args.short_documents else 3):n - 1] = 1
                lo, hi = context_shard_bounds(n, rank, size)
                samples.append({
                    "input_ids": ids[doc:doc + 1, :n],
                    "loss_mask": mask[doc:doc + 1, :n],
                    "hidden_states": auxiliary[doc:doc + 1, lo:hi],
                    "last_hidden_states": teacher[doc:doc + 1, lo:hi],
                    "_aspec_work": {"execution_id": f"doc-{doc}"},
                })
            packed = DFlashPackingCollator(
                max_seq_length=64, pad_to_max_length=False,
                context_parallel_rank=rank, context_parallel_size=size,
            )(samples)
            layout, states, local_teacher = EagleTHDLayout.from_batch(
                packed, cp_rank=rank, cp_size=size, group=dist.group.WORLD,
            )
            result = model(
                input_ids=layout.full_ids[None], attention_mask=None,
                target=LazyTarget(local_teacher, head, t2d), loss_mask=layout.full_loss[None],
                hidden_states=states, thd_layout=layout,
            )
            weights = (1.0, 0.8, 0.64)
            actual_loss = sum(w * loss for w, loss in zip(weights, result[0], strict=True))
            actual_loss.backward()
            if size > 1:
                for parameter in draft.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(size)
            if args.reference_backend == "sdpa":
                expected = reference(
                    input_ids=padding(ids, left=False), attention_mask=attention,
                    target=(compute_target_p_padded(padding(teacher, left=False), head, t2d, mask, 3)
                            if t2d is not None else compute_lazy_target_padded(padding(teacher, left=False), head, 3)),
                    loss_mask=mask, hidden_states=auxiliary,
                )
            else:
                full_layout, full_auxiliary, full_teacher = EagleTHDLayout.from_batch({
                    "input_ids": ids, "attention_mask": attention, "loss_mask": mask,
                    "hidden_states": auxiliary, "last_hidden_states": teacher,
                })
                expected = reference(
                    input_ids=full_layout.full_ids[None], attention_mask=None,
                    target=LazyTarget(full_teacher, head, t2d), loss_mask=full_layout.full_loss[None],
                    hidden_states=full_auxiliary, thd_layout=full_layout,
                )
            ref_loss = sum(w * loss for w, loss in zip(weights, expected[0], strict=True))
            ref_loss.backward()
            actual_params = {n: p for n, p in draft.named_parameters() if p.requires_grad}
            ref_params = dict(ref_draft.named_parameters())
            gradients = difference([p.grad for p in actual_params.values()],
                                   [ref_params[n].grad for n in actual_params])
            loss_stats = difference(result[1], expected[1])
            per_tensor = {n: difference([p.grad], [ref_params[n].grad]) for n, p in actual_params.items()}
            gates = []
            if args.gate_diagnostics and kind != "forward_kl":
                for depth, (ah, rh) in enumerate(zip(captured, ref_captured, strict=True)):
                    rh = torch.cat([row[:n] for row, n in zip(rh, lengths, strict=True)])
                    valid = layout.depth_inputs(depth)[1].flatten().bool()
                    th = layout.teacher_at(local_teacher, depth).reshape(-1, 32)[valid]
                    gates.append(gate_probe(ah.reshape(-1, 32)[valid], rh[valid],
                                            (th, head), draft, kind, model.lk_eta))
            optimizer.step()
            ref_optimizer.step()
            if args.eagle4_moe and args.eagle4_stage == "draft":
                draft.midlayer.mlp.gate.update_bias()
                # The reference repeats the complete workload on each rank.
                # Its all-reduced load is uniformly scaled; the mean-minus-load
                # sign is unchanged. Integer division would truncate real load.
                ref_draft.midlayer.mlp.gate.update_bias()
                torch.testing.assert_close(draft.midlayer.mlp.gate.e_score_correction_bias,
                                           ref_draft.midlayer.mlp.gate.e_score_correction_bias, atol=0, rtol=0)
            state = difference([p for p in actual_params.values()], [ref_params[n] for n in actual_params])
            master = difference(optimizer.fp32_params, ref_optimizer.fp32_params)
            moments = {
                key: difference(
                    [optimizer.state[p][key] for p in optimizer.fp32_params],
                    [ref_optimizer.state[p][key] for p in ref_optimizer.fp32_params],
                ) for key in ("exp_avg", "exp_avg_sq")
            }
            record = {"kind": kind, "step": step + 1, "cp_size": size,
                      "loss": loss_stats, "gradients": gradients, "parameters": state,
                      "per_tensor_gradients": per_tensor, "master": master, "moments": moments,
                      "gate_diagnostics": gates,
                      "counts_equal": torch.equal(torch.stack(result[3]), torch.stack(expected[3]))}
            if not record["counts_equal"]:
                raise AssertionError(record)
            records.append(record)
            if rank == 0:
                print({k: v for k, v in record.items() if k != "per_tensor_gradients"}, flush=True)
            # Both optimizers make their finite-update collective on every rank.
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
            dist.barrier()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"synthetic_data": True, "mla": args.mla, "eagle4_stage": args.eagle4_stage,
                                          "eagle4_moe": args.eagle4_moe, "steps": args.steps,
                                          "cp_size": size, "common_states": args.common_states,
                                          "reference_backend": args.reference_backend,
                                          "draft_vocab_size": args.draft_vocab_size,
                                          "query_heads": args.query_heads, "kv_heads": args.kv_heads,
                                          "short_documents": args.short_documents,
                                          "gate_diagnostics": args.gate_diagnostics, "records": records}, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
