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

"""EAGLE4 uses existing trainer ownership, optimizer, evaluation and checkpoints."""
from types import SimpleNamespace

import torch
import torch.distributed as dist

from aspec.models.draft.eagle4_moe import Eagle4MoE
from aspec.models.eagle3 import LazyTarget
from aspec.models.eagle4 import Eagle4TrainingModel
from aspec.training import checkpoint
from aspec.training.eagle3_trainer import Eagle3Trainer
from aspec.utils.tensor import padding


class Eagle4Trainer(Eagle3Trainer):
    def _prepare_algorithm_training(self, draft):
        resume = checkpoint.resolve_resume_model_dir(self.args)
        draft.configure_training_stage()
        if draft.candidate_stage:
            if getattr(self.args, "continual_training", False) and resume:
                raise ValueError("Candidate-stage load_path is exact resume; use initial_draft_model_path for stage initialization")
            if resume is None:
                if not getattr(self.args, "initial_draft_model_path", None):
                    raise ValueError("Candidate training requires a pretrained Draft or same-stage checkpoint")
                if not draft.lm_head.weight.is_meta and int(draft.candidate_head_steps) == 0:
                    draft.initialize_candidate_head(device=torch.cuda.current_device())
        else:
            if getattr(self.args, "initial_draft_model_path", None):
                draft.candidate_head_steps.zero_()
            if getattr(self.args, "freeze_lm_head", False):
                draft.freeze_lm_head()

    def _create_training_model(self, draft_model):
        return Eagle4TrainingModel(
            draft_model, length=self.args.ttt_length, attention_backend=self.args.attention_backend,
            gradient_checkpointing=getattr(self.args, "gradient_checkpointing", False) and not draft_model.candidate_stage,
            loss_type=getattr(self.args, "loss_type", "forward_kl"), lk_eta=getattr(self.args, "lk_eta", 3.0),
        )

    def _init_target_lm_head(self, target_model_path):
        if self.draft_model.candidate_stage:
            # Retrieval teacher is the frozen Draft. Do not load/copy a Target LM head.
            self.target_lm_head = SimpleNamespace(lm_head=SimpleNamespace(weight=None), norm=None)
        else:
            super()._init_target_lm_head(target_model_path)

    def _forward(self, batch):
        if not self.draft_model.candidate_stage:
            return super()._forward(batch)
        if self.args.attention_backend == "fa4":
            from aspec.models.ops.eagle_thd import EagleTHDLayout
            layout, auxiliary, _ = EagleTHDLayout.from_batch(
                batch, cp_rank=self.cp_rank, cp_size=self.cp_size, group=self.cp_group,
                teacher_required=False,
            )
            return self.model(
                input_ids=layout.full_ids.unsqueeze(0), attention_mask=None,
                target=LazyTarget(None, None), loss_mask=layout.full_loss.unsqueeze(0),
                hidden_states=auxiliary, thd_layout=layout,
            )
        ids = padding(batch["input_ids"], left=False).cuda()
        mask = batch["loss_mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        return self.model(
            input_ids=ids, attention_mask=batch["attention_mask"].cuda(),
            target=LazyTarget(None, None), loss_mask=mask.cuda(),
            hidden_states=batch["hidden_states"].cuda(),
            position_ids=batch["position_ids"].cuda() if batch.get("position_ids") is not None else None,
        )

    def get_feature_contract_status(self):
        status = super().get_feature_contract_status()
        if self.draft_model.candidate_stage:
            metadata = dict(status.get("execution_metadata") or {})
            metadata.update(
                draft_attention_backward_required=False,
                draft_body_frozen=all(not p.requires_grad for name, p in self.draft_model.named_parameters()
                                      if not name.startswith("efficient_lm_head")),
                draft_attention_backward=["not_required_frozen_backbone"],
            )
            status = {**status, "execution_metadata": metadata}
        return status

    def _after_optimizer_step(self, all_step_metrics, grad_norm):
        draft = self.draft_model
        finite = torch.isfinite(grad_norm) if grad_norm is not None else False
        if draft.candidate_stage:
            counts = torch.stack([m["acc_counts"].sum() for m in all_step_metrics]).sum()
            # Supervision can be empty on one DP owner while synchronized
            # gradients still update its head. Replicated proof must agree.
            if dist.is_initialized():
                dist.all_reduce(counts, group=self.dp_group)
            draft.candidate_head_steps.add_(((counts > 0) & torch.as_tensor(finite, device=counts.device)).long())
        elif isinstance(draft.midlayer.mlp, Eagle4MoE):
            draft.midlayer.mlp.gate.update_bias(apply_update=finite, group=self.dp_group)

    def _metric_semantics(self, result, namespace):
        if self.draft_model.candidate_stage:
            result.pop(f"{namespace}/simulated_acc_len", None)
            result[f"{namespace}/candidate_top1_recall"] = result[f"{namespace}/avg_acc"]
            result[f"{namespace}/candidate_teacher_mass"] = result.pop(f"{namespace}/avg_alpha")
            for key in list(result):
                if key.startswith(f"{namespace}/alpha_"):
                    result[key.replace("/alpha_", "/candidate_teacher_mass_")] = result.pop(key)
        return result
