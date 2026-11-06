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

"""Native MTP uses the existing optimizer/checkpoint/ownership training path."""

from types import SimpleNamespace

from aspec.models.mtp import MTPModel
from aspec.training import checkpoint
from aspec.training.eagle3_trainer import Eagle3Trainer


class MTPTrainer(Eagle3Trainer):
    def _create_training_model(self, draft_model):
        return MTPModel(draft_model, length=self.args.mtp_steps,
                        gradient_checkpointing=self.args.gradient_checkpointing)

    def _load_algorithm_weights(self, draft_model, target_model_path):
        if not (getattr(self.args, "initial_draft_model_path", None)
                or checkpoint.resolve_resume_model_dir(self.args)):
            draft_model.initialize_native_weights(target_model_path)

    def _init_target_lm_head(self, target_model_path):
        # CE needs no teacher probability head. Reuse the already frozen native
        # output head rather than allocating a second vocabulary-sized matrix.
        if (getattr(self.args, "initial_draft_model_path", None)
                or checkpoint.resolve_resume_model_dir(self.args)):
            self.draft_model.validate_target_binding(
                target_model_path, embedding_key=self.args.embedding_key,
                lm_head_key=getattr(self.args, "lm_head_key", "lm_head.weight"),
            )
        self.target_lm_head = SimpleNamespace(lm_head=self.draft_model.lm_head, norm=None)

    def _forward(self, batch):
        if self.args.attention_backend == "fa4":
            from aspec.models.ops.eagle_thd import EagleTHDLayout

            layout, hidden, _ = EagleTHDLayout.from_batch(
                batch, cp_rank=self.cp_rank, cp_size=self.cp_size,
                group=self.cp_group, teacher_required=False,
            )
            return self.model(input_ids=None, hidden_states=hidden, loss_mask=None,
                              attention_mask=None, thd_layout=layout)
        return self.model(
            input_ids=batch["input_ids"].cuda(),
            hidden_states=batch["hidden_states"].cuda(),
            loss_mask=batch["loss_mask"].squeeze(-1).cuda()
            if batch["loss_mask"].ndim == 3 else batch["loss_mask"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
        )
