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

"""Candidate retrieval training, reimplemented from public EAGLE4 stage semantics.

Teacher: frozen *Draft* exact logits, restricted to its Top-K and renormalized.
Student: full-vocabulary low-rank distribution. No Target KL or joint backbone
update is substituted for retrieval training. This is not the unpublished recipe.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def candidate_retrieval_terms(hidden, full_head, down, up, *, teacher_top_k, num_candidates, chunk_size=128):
    h = hidden.detach().reshape(-1, hidden.shape[-1])
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Candidate loss chunk size must be positive")
    if not (0 < teacher_top_k <= full_head.shape[0] and 0 < num_candidates <= full_head.shape[0]):
        raise ValueError("Candidate counts exceed vocabulary")
    def chunk(states, down_weight, up_weight):
        with torch.no_grad():
            logits = F.linear(states, full_head).float()
            selected, teacher_ids = logits.topk(teacher_top_k, -1)
            teacher_p = F.softmax(selected, -1)
        scores = F.linear(F.linear(states, down_weight), up_weight).float()
        log_q = scores.log_softmax(-1).gather(-1, teacher_ids)
        loss = (teacher_p * (teacher_p.clamp_min(torch.finfo(torch.float32).tiny).log() - log_q)).sum()
        with torch.no_grad():
            candidates = scores.topk(num_candidates, -1).indices
            recall = (candidates == teacher_ids[:, :1]).any(-1).float().sum()
            mass = (logits.gather(-1, candidates) - logits.logsumexp(-1, keepdim=True)).exp().sum()
        return loss, recall, mass
    total = (down.sum() + up.sum()) * 0.0
    recall = total.detach().clone()
    mass = recall.clone()
    for states in h.split(chunk_size):
        values = (checkpoint(chunk, states, down, up, use_reentrant=False)
                  if torch.is_grad_enabled() else chunk(states, down, up))
        total = total + values[0]
        recall = recall + values[1]
        mass = mass + values[2]
    return total, recall, torch.tensor(h.shape[0], device=h.device, dtype=torch.float32), mass
