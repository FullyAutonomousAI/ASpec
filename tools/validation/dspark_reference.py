# MIT License
#
# Copyright (c) 2026 The DeepSpec Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Small DSpark objective oracle, adapted from official DeepSpec at
0a03e19e1c57000d5d3d291fa57109944eba6b6a.

Sources: modeling/dspark/qwen3/modeling.py (labels/teacher/Markov),
common.py (eval mask), loss.py (CE/L1/confidence/decay/reduction).
Metrics and distributed plumbing are omitted; callers pool additive terms.
Document isolation is the explicit packing extension, not an upstream claim.
This file must not import ASpec's loss or label helpers.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F


def official_dspark_terms(hidden, draft, batch, anchors, keep, block_size, gamma=4.0):
    bsz, seq_len = batch["input_ids"].shape
    hidden = hidden.reshape(bsz, anchors.shape[1], block_size, -1)
    label_indices = anchors[..., None] + torch.arange(1, block_size + 1, device=anchors.device)
    safe = label_indices.clamp(max=seq_len - 1)
    safe = torch.where(keep[..., None], safe, torch.zeros_like(safe))
    target_ids = batch["input_ids"][:, None].expand(-1, anchors.shape[1], -1).gather(2, safe)
    target_mask = batch["loss_mask"][:, None].expand(-1, anchors.shape[1], -1).gather(2, safe)
    valid = (label_indices < seq_len) & (target_mask > 0.5) & keep[..., None]
    if batch.get("ctx_doc_ids") is not None:
        docs = batch["ctx_doc_ids"]
        target_docs = docs[:, None].expand(-1, anchors.shape[1], -1).gather(2, safe)
        valid &= target_docs == docs.gather(1, anchors)[..., None]
    eval_mask = valid.int().cumprod(-1).bool()
    weights = eval_mask.float()
    if gamma is not None and gamma > 0:
        weights = weights * torch.exp(
            -torch.arange(block_size, device=anchors.device).float() / gamma
        )
    previous = torch.cat(
        (batch["input_ids"].gather(1, anchors)[..., None], target_ids[..., :-1]), -1
    )
    logits = F.linear(hidden, batch["lm_head_weight"])
    prev_embeddings = None
    if draft.markov_head is not None:
        prev_embeddings = F.embedding(previous, draft.markov_head.markov_w1.weight)
        logits = logits + F.linear(prev_embeddings, draft.markov_head.markov_w2.weight)
    vocab = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none"
    ).reshape_as(weights)
    denominator = weights.sum()
    ce_num = (ce * weights).sum()
    l1_num = logits.new_zeros((), dtype=torch.float32)
    conf_num = logits.new_zeros((), dtype=torch.float32)
    teacher_logits = None
    prediction = None
    if batch.get("last_hidden_states") is not None:
        teacher_idx = (safe - 1).clamp(min=0)
        teacher_hidden = (
            batch["last_hidden_states"][:, None]
            .expand(-1, anchors.shape[1], -1, -1)
            .gather(2, teacher_idx[..., None].expand(-1, -1, -1, hidden.shape[-1]))
        )
        teacher_logits = F.linear(teacher_hidden, batch["lm_head_weight"])
        l1_per_token = (
            (logits.float().softmax(-1) - teacher_logits.float().softmax(-1)).abs().sum(-1)
        )
        l1_num = (l1_per_token * weights).sum()
        if draft.confidence_head is not None:
            features = hidden
            if draft.confidence_head_with_markov:
                features = torch.cat((hidden, prev_embeddings.to(hidden.dtype)), -1)
            prediction = (
                F.linear(
                    features,
                    draft.confidence_head.proj.weight,
                    draft.confidence_head.proj.bias,
                )
                .squeeze(-1)
                .float()
            )
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0).detach()
            conf_num = (
                F.binary_cross_entropy_with_logits(prediction, accept_rate, reduction="none")
                * weights
            ).sum()
    return {
        "draft_logits": logits,
        "aligned_target_logits": teacher_logits,
        "confidence_pred": prediction,
        "block_keep_mask": keep,
        "ce_loss": (ce_num, denominator),
        "l1_loss": (l1_num, denominator),
        "confidence_loss": (conf_num, denominator),
        "target_ids": target_ids,
        "teacher_indices": (safe - 1).clamp(min=0),
        "previous_ids": previous,
        "eval_mask": eval_mask,
    }


def official_loss(terms, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=1.0):
    # DeepSpec adds epsilon rather than clamping; retain that in the oracle.
    return sum(
        alpha * terms[name][0] / (terms[name][1] + 1e-6)
        for name, alpha in (
            ("ce_loss", ce_alpha),
            ("l1_loss", l1_alpha),
            ("confidence_loss", confidence_alpha),
        )
    )


def check_unmodified_official_loss(source_path, terms):
    """Execute the pinned official function bodies, not this adapted oracle."""
    tree = ast.parse(Path(source_path).read_text())
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "F": F,
        "Optional": Optional,
        "DSparkForwardOutput": SimpleNamespace,
        "add_metric": lambda *args, **kwargs: None,
    }
    exec(compile(functions, str(source_path), "exec"), namespace)
    outputs = SimpleNamespace(
        **{
            key: terms[key]
            for key in (
                "draft_logits",
                "aligned_target_logits",
                "confidence_pred",
                "block_keep_mask",
                "target_ids",
                "eval_mask",
            )
        }
    )
    actual, has_confidence = namespace["_collect_local_terms"](
        outputs=outputs, loss_decay_gamma=4.0, l1_loss_alpha=0.9
    )
    for name in ("ce_loss", "l1_loss", "confidence_loss"):
        torch.testing.assert_close(actual[name + "_num"], terms[name][0])
        torch.testing.assert_close(actual[name + "_den"], terms[name][1])
    official = namespace["_build_loss"](
        loss_terms=actual,
        global_denominators={key: value for key, value in actual.items() if key.endswith("_den")},
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
        has_confidence=has_confidence,
        world_size=1,
    )
    torch.testing.assert_close(official, official_loss(terms))
