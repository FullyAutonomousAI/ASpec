# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from aspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
    compiled_forward_kl_loss_from_logits,
    compiled_lk_alpha_loss,
    compiled_lk_alpha_loss_from_hs,
    compiled_lk_alpha_loss_from_logits,
    compiled_lk_lambda_loss,
    compiled_lk_lambda_loss_from_hs,
    compiled_lk_lambda_loss_from_logits,
    compiled_sum_forward_kl_loss_from_hs,
)
from aspec.utils.distributed import (
    get_draft_sp_group,
    get_sp_ring_rank,
    get_sp_ulysses_group,
)
from aspec.utils.tensor import padding


@dataclass
class PrecomputedTarget:
    """Pre-computed target probabilities (used with vocab pruning)."""

    target_p_padded: torch.Tensor  # (B, T + length, V_draft), renormalized over the draft vocab
    position_mask: Optional[torch.Tensor] = None  # (B, T)
    # (B, T + length) draft-vocab probability mass under the *true*, full-vocab
    # softmax (S = sum_{v in draft} p(v)). LK losses need the true, un-renormalized
    # target probabilities (target_p_padded * coverage_padded) so that out-of-draft
    # mass contributes 0 rather than being redistributed — see arXiv:2602.23881 §4.4.
    # None when not computed (e.g. hand-built targets in tests); treated as 1.
    coverage_padded: Optional[torch.Tensor] = None


@dataclass
class LazyTarget:
    """Deferred target computation to avoid materializing (B, T, V_full)."""

    hidden_states_padded: torch.Tensor  # (B, T + length, D)
    lm_head_weight: torch.Tensor  # (V_full, D)
    t2d: Optional[torch.Tensor] = None  # full-vocabulary membership, checkpoint-bound
    pruned_weight: Optional[torch.Tensor] = None


@dataclass
class CachedLogitsTarget:
    """Full-vocabulary Target logits reused by every recurrent depth."""

    target_logits_padded: torch.Tensor  # (B, T + length, V_full)


class Eagle3Model(nn.Module):
    def __init__(
        self,
        draft_model,
        length: int = 7,
        attention_backend="sdpa",
        gradient_checkpointing: bool = False,
        loss_type: str = "forward_kl",
        lk_eta: float = 3.0,
        loss_gradient_checkpointing: Optional[bool] = None,
    ):
        super().__init__()
        supported_loss_types = {"forward_kl", "lk_alpha", "lk_lambda"}
        if loss_type not in supported_loss_types:
            raise ValueError(
                f"Unsupported Eagle3 loss_type={loss_type!r}; "
                f"expected one of {sorted(supported_loss_types)}"
            )
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing
        self.loss_gradient_checkpointing = (
            gradient_checkpointing
            if loss_gradient_checkpointing is None
            else loss_gradient_checkpointing
        )
        self.loss_type = loss_type
        self.lk_eta = lk_eta
        self.vocab_pruning = draft_model.vocab_size != draft_model.target_vocab_size
        self._usp_sp_group = get_draft_sp_group() if attention_backend == "usp" else None
        self._usp_ulysses_group = get_sp_ulysses_group() if attention_backend == "usp" else None
        self._usp_ulysses_world_size = (
            dist.get_world_size(self._usp_ulysses_group)
            if self._usp_ulysses_group is not None
            else 1
        )

    @staticmethod
    def _coverage_flat(
        target: PrecomputedTarget,
        idx: int,
        seq_length: int,
        tp_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Flattened (B*T,) coverage for this depth's slice of ``target``.

        Falls back to all-ones (no rescaling) when ``coverage_padded`` wasn't
        computed, which is only the case for hand-built targets that don't
        involve vocab pruning.
        """
        if target.coverage_padded is None:
            return tp_flat.new_ones(tp_flat.shape[0])
        coverage_step = target.coverage_padded[:, idx : idx + seq_length]
        return coverage_step.reshape(-1)

    def _usp_reduced_mean_alpha(
        self,
        alpha_probe_fn,
        probe_args: tuple,
    ) -> torch.Tensor:
        """SP-group-reduced mean acceptance rate for this depth.

        The LK^lambda schedule is defined per depth using alpha aggregated
        across the *full* batch and sequence (arXiv:2602.23881), but under USP
        the sequence is sharded across ranks, so a single rank's local alpha
        only covers its shard. This runs a cheap no-grad probe (reusing the
        LK^alpha loss fn, which already computes alpha_sum/count) and
        all-reduces across the SP group before lambda is formed elsewhere.

        Every rank in the SP group must call this (or the zero-contribution
        path in ``_calculate_loss``'s empty-``valid_idx`` branch) exactly once
        per depth when ``loss_type == "lk_lambda"`` under USP, so the
        collective never hangs waiting for a rank that skipped it.
        """
        with torch.no_grad():
            _, _, probe_count, probe_alpha_sum = alpha_probe_fn(*probe_args)
            stats = torch.stack((probe_alpha_sum.float(), probe_count.float()))
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self._usp_sp_group)
            reduced_alpha_sum, reduced_count = stats.unbind()
            return (reduced_alpha_sum / reduced_count.clamp_min(1.0)).detach()


    def _calculate_pruned_loss(
        self, hidden_states, target, mask, idx, seq_length, norm_weight,
        lm_head_weight, norm_eps, lm_head_up,
    ):
        """Bound teacher/logit workspace independently of sequence length.

        Match the retained TorchSpec pruning rule: supervise only when the
        full Target argmax belongs to the draft vocabulary. KL renormalizes
        within that vocabulary; LK acceptance/TV retain the original mass.
        Checkpoint each chunk including its teacher projection, so backward
        does not retain a sequence-by-vocabulary tensor.
        """
        hs = hidden_states.reshape(-1, hidden_states.shape[-1])
        teacher = target.hidden_states_padded[:, idx:idx + seq_length].reshape(
            -1, target.lm_head_weight.shape[-1],
        )
        valid = mask.flatten().nonzero().flatten()
        chunk_size = 128
        kept, normalizers = [], []
        with torch.no_grad():
            for indices in valid.split(chunk_size):
                logits = F.linear(teacher.index_select(0, indices), target.lm_head_weight)
                eligible = target.t2d[logits.argmax(-1)]
                kept.append(indices[eligible])
                normalizers.append(torch.logsumexp(logits.float(), -1)[eligible])
            valid = torch.cat(kept) if kept else valid
            normalizers = torch.cat(normalizers) if normalizers else hs.new_empty(0).float()
        indices = torch.arange(min(chunk_size, valid.numel()), device=hs.device)
        def chunk_loss(states, teachers, full_lse, norm, head, head_up, mean_alpha):
            with torch.no_grad():
                logits = F.linear(teachers, target.pruned_weight)
                probs = F.softmax(logits.float(), -1)
                coverage = (torch.logsumexp(logits.float(), -1) - full_lse).exp()
            chunk_indices = indices[:states.shape[0]]
            args = (states, probs, chunk_indices, norm, head, head_up, norm_eps)
            if self.loss_type == "lk_alpha":
                return compiled_lk_alpha_loss(*args, coverage)
            if self.loss_type == "lk_lambda":
                return compiled_lk_lambda_loss(*args, coverage, mean_alpha, self.lk_eta)
            loss, correct, count = compiled_forward_kl_loss(*args)
            return loss, correct, count, count.new_zeros(())

        mean_alpha = None
        if self.loss_type == "lk_lambda":
            # One schedule per depth, not per chunk or CP owner.
            stats = torch.zeros(2, device=hs.device, dtype=torch.float32)
            with torch.no_grad():
                for offset in range(0, valid.numel(), chunk_size):
                    selected = valid[offset:offset + chunk_size]
                    teachers = teacher.index_select(0, selected)
                    logits = F.linear(teachers, target.pruned_weight)
                    probs = F.softmax(logits.float(), -1)
                    coverage = (torch.logsumexp(logits.float(), -1)
                                - normalizers[offset:offset + chunk_size]).exp()
                    probe = compiled_lk_alpha_loss(
                        hs.index_select(0, selected), probs, indices[:selected.numel()],
                        norm_weight, lm_head_weight, lm_head_up, norm_eps, coverage,
                    )
                    stats += torch.stack((probe[3].float(), probe[2].float()))
                if self._usp_sp_group is not None:
                    dist.all_reduce(stats, group=self._usp_sp_group)
                mean_alpha = (stats[0] / stats[1].clamp_min(1)).detach()
        if not valid.numel():
            zero = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad) * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()
        totals = None
        for offset in range(0, valid.numel(), chunk_size):
            selected = valid[offset:offset + chunk_size]
            args = (hs.index_select(0, selected), teacher.index_select(0, selected),
                    normalizers[offset:offset + chunk_size], norm_weight,
                    lm_head_weight, lm_head_up, mean_alpha)
            if self.training and torch.is_grad_enabled():
                result = torch_checkpoint(chunk_loss, *args, use_reentrant=False)
            else:
                result = chunk_loss(*args)
            totals = result if totals is None else tuple(a + b for a, b in zip(totals, result, strict=True))
        return totals

    def _calculate_loss(
        self,
        hidden_states: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget, CachedLogitsTarget],
        mask: torch.Tensor,
        idx: int,
        seq_length: int,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm_eps: float,
        lm_head_up: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (loss_sum, correct_sum, count, alpha_sum).

        ``alpha_sum`` (the LK acceptance-rate numerator) is a detached zero for
        ``loss_type="forward_kl"``, which does not compute it.
        """
        if isinstance(target, LazyTarget) and target.t2d is not None:
            return self._calculate_pruned_loss(
                hidden_states, target, mask, idx, seq_length, norm_weight,
                lm_head_weight, norm_eps, lm_head_up,
            )
        needs_usp_lambda_reduce = self.loss_type == "lk_lambda" and self._usp_sp_group is not None
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        if valid_idx.numel() == 0:
            if needs_usp_lambda_reduce:
                # This rank has nothing to contribute at this depth, but must
                # still participate in the SP-wide alpha all-reduce (with a
                # zero contribution) so peer ranks with valid tokens at this
                # depth don't hang waiting for it — see
                # ``_usp_reduced_mean_alpha``.
                zero_stats = torch.zeros(2, device=hidden_states.device, dtype=torch.float32)
                dist.all_reduce(zero_stats, op=dist.ReduceOp.SUM, group=self._usp_sp_group)
            # FSDP requires every trainable param to participate in gradient
            # all-reduce/reduce-scatter.
            total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
            zero = total * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()
        # Important as it prevents recompilation.
        torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        is_lk = self.loss_type in ("lk_alpha", "lk_lambda")

        if isinstance(target, PrecomputedTarget):
            target_p_step = target.target_p_padded[:, idx : idx + seq_length, :]
            tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])
            args = (hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, lm_head_up, norm_eps)
            if self.loss_type == "lk_alpha":
                fn = compiled_lk_alpha_loss
                args = args + (self._coverage_flat(target, idx, seq_length, tp_flat),)
            elif self.loss_type == "lk_lambda":
                coverage_flat = self._coverage_flat(target, idx, seq_length, tp_flat)
                probe_args = (
                    hs_flat,
                    tp_flat,
                    valid_idx,
                    norm_weight,
                    lm_head_weight,
                    lm_head_up,
                    norm_eps,
                    coverage_flat,
                )
                external_mean_alpha = (
                    self._usp_reduced_mean_alpha(compiled_lk_alpha_loss, probe_args)
                    if needs_usp_lambda_reduce
                    else None
                )
                fn = compiled_lk_lambda_loss
                args = args + (coverage_flat, external_mean_alpha, self.lk_eta)
            else:
                fn = compiled_forward_kl_loss
            if self.loss_gradient_checkpointing and self.training:
                result = torch_checkpoint(fn, *args, use_reentrant=False)
            else:
                result = fn(*args)
        elif isinstance(target, CachedLogitsTarget):
            target_logits_step = target.target_logits_padded[:, idx : idx + seq_length, :]
            target_logits_flat = target_logits_step.reshape(-1, target_logits_step.shape[-1])
            args = (
                hs_flat,
                target_logits_flat,
                valid_idx,
                norm_weight,
                lm_head_weight,
                lm_head_up,
                norm_eps,
            )
            if self.loss_type == "lk_alpha":
                fn = compiled_lk_alpha_loss_from_logits
            elif self.loss_type == "lk_lambda":
                external_mean_alpha = (
                    self._usp_reduced_mean_alpha(compiled_lk_alpha_loss_from_logits, args)
                    if needs_usp_lambda_reduce
                    else None
                )
                args = args + (external_mean_alpha, self.lk_eta)
                fn = compiled_lk_lambda_loss_from_logits
            else:
                fn = compiled_forward_kl_loss_from_logits
            if self.loss_gradient_checkpointing and self.training:
                result = torch_checkpoint(fn, *args, use_reentrant=False)
            else:
                result = fn(*args)
        else:
            ths_flat = target.hidden_states_padded[:, idx : idx + seq_length, :].reshape(
                -1, target.lm_head_weight.shape[-1]
            )
            args = (
                hs_flat,
                ths_flat,
                valid_idx,
                norm_weight,
                lm_head_weight,
                lm_head_up,
                target.lm_head_weight,
                norm_eps,
            )
            use_sum_lazy_loss = self.attention_backend == "usp"
            if self.loss_type == "lk_alpha":
                fn = compiled_lk_alpha_loss_from_hs
            elif self.loss_type == "lk_lambda":
                external_mean_alpha = (
                    self._usp_reduced_mean_alpha(compiled_lk_alpha_loss_from_hs, args)
                    if needs_usp_lambda_reduce
                    else None
                )
                args = args + (external_mean_alpha, self.lk_eta)
                fn = compiled_lk_lambda_loss_from_hs
            else:
                fn = (
                    compiled_sum_forward_kl_loss_from_hs
                    if use_sum_lazy_loss
                    else compiled_forward_kl_loss_from_hs
                )
            if self.loss_gradient_checkpointing and self.training:
                result = torch_checkpoint(fn, *args, use_reentrant=False)
            else:
                result = fn(*args)

        if is_lk:
            return result
        loss_sum, correct_sum, count = result
        return loss_sum, correct_sum, count, count.new_zeros(())

    def _forward_thd(self, hidden_states, target, layout):
        if not isinstance(target, LazyTarget):
            raise ValueError("THD EAGLE requires lazy Target features")
        self._usp_sp_group = layout.group if layout.cp_size > 1 else None
        norm, head, eps = self.draft_model.get_lm_head_params()
        head_up = self.draft_model.get_lm_head_up_weight()
        hidden_states = self.draft_model.project_hidden_states(hidden_states)
        if target.hidden_states_padded is not None:
            layout.prepare_teacher_halo(target.hidden_states_padded, self.length)
        keys = values = None
        results = [[], [], [], [], []]
        for depth in range(self.length):
            ids, mask = layout.depth_inputs(depth)
            inputs = self.draft_model.embed_input_ids(ids).to(hidden_states.dtype)
            def backbone(embeds, states, old_keys, old_values):
                return self.draft_model.backbone(
                    input_embeds=embeds, hidden_states=states, attention_mask=layout,
                    position_ids=layout.rope.unsqueeze(0), cache_keys=old_keys,
                    cache_values=old_values, use_cache=True,
                )
            if self.gradient_checkpointing and self.training:
                hidden_states, keys, values = torch_checkpoint(
                    backbone, inputs, hidden_states, keys, values, use_reentrant=False,
                )
            else:
                hidden_states, keys, values = backbone(inputs, hidden_states, keys, values)
            self.draft_model.commit_router_load()
            teacher = LazyTarget(layout.teacher_at(target.hidden_states_padded, depth) if target.hidden_states_padded is not None else None,
                                 target.lm_head_weight, target.t2d, target.pruned_weight)
            numerator, correct, count, alpha = self._calculate_loss(
                hidden_states=hidden_states, target=teacher, mask=mask, idx=0,
                seq_length=layout.local_length, norm_weight=norm, lm_head_weight=head,
                norm_eps=eps, lm_head_up=head_up,
            )
            # Even an all-masked owner participates in the same CP backward.
            numerator = numerator + hidden_states.sum() * 0.0
            stats = torch.stack((numerator.detach().float(), correct.detach().float(),
                                 count.detach().float(), alpha.detach().float()))
            if layout.cp_size > 1:
                dist.all_reduce(stats, group=layout.group)
            denominator = stats[2].clamp_min(1.0)
            # FSDP averages over DP*CP, while the objective averages each
            # microbatch over its CP union. Preserve the existing DP convention.
            results[0].append(numerator / denominator * layout.cp_size)
            results[1].append(stats[0] / denominator)
            results[2].append(stats[1] / denominator)
            results[3].append(stats[2])
            results[4].append(stats[3] / denominator)
            hidden_states = self.draft_model.next_depth_hidden(hidden_states)
        return tuple(results)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget, CachedLogitsTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        thd_layout=None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        if isinstance(target, LazyTarget) and target.t2d is not None and target.pruned_weight is None:
            target.pruned_weight = target.lm_head_weight[target.t2d].detach()
        if thd_layout is not None:
            return self._forward_thd(hidden_states, target, thd_layout)
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        norm_weight, lm_head_weight, norm_eps = self.draft_model.get_lm_head_params()
        lm_head_up = self.draft_model.get_lm_head_up_weight()
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if self.attention_backend == "usp":
            usp_chunk_size = seq_length - self.length
            if usp_chunk_size <= 0:
                raise ValueError(
                    f"USP local seq_length ({seq_length}) must be larger than ttt_length ({self.length})"
                )
            if position_ids is None:
                device = hidden_states.device
                ring_chunk_size = usp_chunk_size * self._usp_ulysses_world_size
                position_start = get_sp_ring_rank() * ring_chunk_size + past_key_values_length
                position_ids = torch.arange(
                    position_start,
                    position_start + ring_chunk_size,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
        elif position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        if isinstance(target, PrecomputedTarget) and target.position_mask is not None:
            mask = target.position_mask
        else:
            mask = loss_mask

        plosses = []
        vlosses = []
        acces = []
        acc_counts = []
        alphas = []
        cache_keys = None
        cache_values = None

        input_ids = input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1)

        for idx in range(self.length):
            is_last = idx == self.length - 1

            step_input_ids = input_ids
            step_hidden_states = hidden_states
            step_attention_mask = attention_mask
            step_position_ids = position_ids
            step_mask = mask
            step_seq_length = seq_length

            if self.attention_backend == "usp":
                step_seq_length = usp_chunk_size
                step_input_ids = input_ids[:, :step_seq_length]
                step_hidden_states = hidden_states[:, :step_seq_length, :]
                step_mask = mask[:, :step_seq_length]
                if attention_mask is not None:
                    step_attention_mask = attention_mask[:, :step_seq_length]
                if position_ids is not None:
                    step_position_ids = position_ids[
                        :, : step_seq_length * self._usp_ulysses_world_size
                    ]

            inputs_embeds = self.draft_model.embed_input_ids(step_input_ids)
            inputs_embeds = inputs_embeds.to(step_hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                hidden_states_out, cache_keys, cache_values = torch_checkpoint(
                    self.draft_model.backbone,
                    inputs_embeds,
                    step_hidden_states,
                    step_attention_mask,
                    step_position_ids,
                    cache_keys,
                    cache_values,
                    True,
                    use_reentrant=False,
                )
            else:
                hidden_states_out, cache_keys, cache_values = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=step_hidden_states,
                    attention_mask=step_attention_mask,
                    position_ids=step_position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )

            hidden_states = hidden_states_out
            self.draft_model.commit_router_load()

            local_sum_loss, local_correct, local_count, local_alpha_sum = self._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=step_mask,
                idx=idx,
                seq_length=step_seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
                lm_head_up=lm_head_up,
            )

            # Model takes its own normed hidden states as input for the next step, so apply norm here if needed.
            hidden_states = self.draft_model.next_depth_hidden(hidden_states)

            if self.attention_backend == "usp":
                # A shard can have no local loss tokens while its Ulysses peers do.
                # Keep the zero-loss path connected to this layer's activations so
                # autograd still executes the same sequence-parallel collectives.
                local_sum_loss = local_sum_loss + hidden_states.sum() * 0.0

            loss = local_sum_loss / local_count.clamp_min(1.0)
            metric_loss = loss.detach()
            metric_acc = (local_correct / local_count.clamp_min(1.0)).detach()
            metric_alpha = (local_alpha_sum / local_count.clamp_min(1.0)).detach()

            if self._usp_sp_group is not None:
                reduced_stats = torch.stack(
                    (
                        local_sum_loss.detach().clone().float(),
                        local_correct.detach().clone().float(),
                        local_count.detach().clone().float(),
                        local_alpha_sum.detach().clone().float(),
                    )
                )
                dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM, group=self._usp_sp_group)
                reduced_sum_loss, reduced_correct, reduced_count, reduced_alpha_sum = (
                    reduced_stats.unbind()
                )
                denom = reduced_count.clamp_min(1.0)
                loss = (local_sum_loss / denom).to(loss.dtype)
                metric_loss = (reduced_sum_loss / denom).detach()
                metric_acc = (reduced_correct / denom).to(device=loss.device, dtype=torch.float32)
                metric_alpha = (reduced_alpha_sum / denom).to(
                    device=loss.device, dtype=torch.float32
                )
                metric_count = reduced_count.to(device=loss.device, dtype=torch.float32)
            else:
                metric_count = local_count.detach().float().to(device=loss.device)

            plosses.append(loss)
            vlosses.append(metric_loss)
            acces.append(metric_acc)
            acc_counts.append(metric_count)
            alphas.append(metric_alpha)

            if not is_last:
                input_ids = padding(input_ids, left=False)
                mask = padding(mask, left=False)
        return plosses, vlosses, acces, acc_counts, alphas


@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    length: int,
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    target_lm_head_weight = target_lm_head_weight.detach()
    pruned_weight = target_lm_head_weight[t2d]

    bsz, seq_len, hidden_size = target_hidden_states.shape
    loss_mask_bool = loss_mask.bool()

    valid_flat_idx = loss_mask_bool.reshape(-1).nonzero(as_tuple=True)[0]
    valid_hs = target_hidden_states.reshape(-1, hidden_size)[valid_flat_idx]

    position_mask_flat = torch.zeros(
        bsz * seq_len,
        device=target_hidden_states.device,
        dtype=torch.float,
    )
    # Draft-vocab probability mass under the *true* full-vocab softmax at each
    # valid position (S = sum_{v in draft} p(v)); lets LK losses undo the
    # renormalization below and recover true, un-renormalized target
    # probabilities. See ``PrecomputedTarget.coverage_padded``.
    coverage_flat = torch.zeros(
        bsz * seq_len,
        device=target_hidden_states.device,
        dtype=torch.float,
    )
    for i in range(0, valid_hs.shape[0], chunk_size):
        chunk_hs = valid_hs[i : i + chunk_size]
        chunk_full_logits = F.linear(chunk_hs, target_lm_head_weight)
        chunk_argmax = chunk_full_logits.argmax(-1)
        in_draft = t2d[chunk_argmax]
        position_mask_flat[valid_flat_idx[i : i + chunk_size]] = in_draft.float()

        # Reuse the already-materialized dense `pruned_weight` (boolean-indexed
        # once, above) instead of boolean-masking `chunk_full_logits` by `t2d`
        # here — masking a CUDA tensor by a bool index forces a device-to-host
        # sync to read back the True-count, and that would otherwise happen
        # once per chunk instead of once per call.
        chunk_pruned_logits = F.linear(chunk_hs, pruned_weight)
        full_logsumexp = torch.logsumexp(chunk_full_logits.float(), dim=-1)
        pruned_logsumexp = torch.logsumexp(chunk_pruned_logits.float(), dim=-1)
        coverage_flat[valid_flat_idx[i : i + chunk_size]] = torch.exp(
            pruned_logsumexp - full_logsumexp
        )
    position_mask = position_mask_flat.reshape(bsz, seq_len)
    coverage_padded = F.pad(coverage_flat.reshape(bsz, seq_len), (0, length), value=0.0)

    target_logits_pruned = F.linear(target_hidden_states, pruned_weight)
    target_p = F.softmax(target_logits_pruned.float(), dim=-1)
    target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

    return PrecomputedTarget(target_p_padded, position_mask, coverage_padded)


def compute_lazy_target_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
    t2d: Optional[torch.Tensor] = None,
    pruned_weight: Optional[torch.Tensor] = None,
) -> LazyTarget:
    return LazyTarget(
        hidden_states_padded=F.pad(target_hidden_states, (0, 0, 0, length), value=0.0),
        lm_head_weight=target_lm_head_weight.detach(),
        t2d=t2d,
        pruned_weight=pruned_weight,
    )


def compute_cached_target_logits_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
) -> CachedLogitsTarget:
    """Project frozen Target features once and reuse their BF16 logits.

    The lazy path repeats this full-vocabulary projection at every recurrent
    depth and again during checkpoint recomputation.  Keeping logits preserves
    the existing FP32 softmax/loss path while bounding the cache to the logits'
    dtype instead of storing full-vocabulary FP32 probabilities.
    """
    with torch.no_grad():
        hidden_states_padded = F.pad(target_hidden_states, (0, 0, 0, length), value=0.0)
        target_logits_padded = F.linear(
            hidden_states_padded,
            target_lm_head_weight.detach(),
        )
    return CachedLogitsTarget(target_logits_padded)
