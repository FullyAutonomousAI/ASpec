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

"""EAGLE4 shares EAGLE's real TTT/THD/CP loop; only retrieval supervision differs."""
import torch

from aspec.models.eagle3 import Eagle3Model
from aspec.models.ops.eagle4_candidate import candidate_retrieval_terms


class Eagle4TrainingModel(Eagle3Model):
    def _calculate_loss(self, hidden_states, target, mask, idx, seq_length,
                        norm_weight, lm_head_weight, norm_eps, lm_head_up):
        if not self.draft_model.candidate_stage:
            return super()._calculate_loss(hidden_states, target, mask, idx, seq_length,
                                          norm_weight, lm_head_weight, norm_eps, lm_head_up)
        draft = self.draft_model
        with torch.no_grad():
            normalized = draft.norm(hidden_states)
            valid = mask.reshape(-1).bool()
            states = normalized.reshape(-1, normalized.shape[-1])[valid]
        policy = draft.eagle4_policy
        return candidate_retrieval_terms(
            states, draft.lm_head.weight, draft.efficient_lm_head_down_proj.weight,
            draft.efficient_lm_head.weight, teacher_top_k=policy["teacher_top_k"],
            num_candidates=policy["num_candidates"], chunk_size=policy["loss_chunk_size"],
        )
