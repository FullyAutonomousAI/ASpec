# SPDX-License-Identifier: BSD-3-Clause
# Adapted from NVIDIA/cudnn-frontend 1.31.0, flex_attention/dispatch.py.
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
# Pradeep Ramani, Tri Dao.
# ASpec modifications Copyright (c) 2026 Ant Group Co., Ltd.
# Internal GQA layouts, autograd ownership, and bounded JIT cache.
"""Opt-in SM90/SM100 BF16 grouped-head backward without physical Q/dO folding."""

import math
from collections import OrderedDict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from threading import Lock

import torch
from torch.autograd.function import once_differentiable

_COMPILED = OrderedDict()
_COMPILE_LOCK = Lock()
_MAX_COMPILED = 128


@lru_cache(maxsize=1)
def _validate_version():
    try:
        installed = version("nvidia-cudnn-frontend")
    except PackageNotFoundError as exc:
        raise RuntimeError("internal GQA backward requires cuDNN Frontend 1.31.0") from exc
    if installed != "1.31.0":
        raise RuntimeError("internal GQA backward is pinned to cuDNN Frontend 1.31.0")


def validate_internal_gqa(query, key, value, mask):
    if any(t.device != query.device for t in (key, value, mask.mask_func)):
        raise ValueError("internal GQA backward requires Q/K/V and mask on the same device")
    if query.dtype != torch.bfloat16:
        raise ValueError("internal GQA backward requires BF16 Q/K/V")
    if query.shape[-2] != 4 * key.shape[-2]:
        raise ValueError("internal GQA backward requires Hq/Hkv=4")
    if query.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("internal GQA backward requires Dqk=Dv=128")
    if not all(t.is_contiguous() for t in (query, key, value)):
        raise ValueError("internal GQA backward requires contiguous Q/K/V")
    if query.shape[-3] == 0 or key.shape[-3] == 0:
        raise ValueError("internal GQA backward requires nonempty Q/K")
    if mask.mask_func.shape[0] != 1:
        raise ValueError("internal GQA backward requires a head-independent mask")
    _validate_version()
    if torch.cuda.get_device_capability(query.device)[0] not in (9, 10):
        raise ValueError("internal GQA backward requires SM90 or SM100-family execution")


def _layout_key(tensor, *, sparse=False):
    if tensor is None:
        return None
    shape = tuple(tensor.shape)
    if sparse and shape and shape[0] > 1:
        # The planner's compact arrays vary in length after anchor resampling.
        # to_cute_block_sparse_tensors marks their layouts dynamic: the leading
        # extent is a runtime value, not a specialization of the compiled MMA.
        # Keep rank, payload dimensions, strides, dtype/device, and 0/1 extents
        # distinct; only erase the changing nontrivial metadata cardinality.
        shape = (None, *shape[1:])
    return (shape, tuple(tensor.stride()), tensor.dtype, tensor.device)


class _BackwardPlan:
    """Immutable metadata; each autograd invocation owns its gradient workspace."""

    def __init__(self, query, key, value, mask):
        from cudnn.flex_attention.dispatch import _block_sparse_runtime_tuple
        from cudnn.flex_attention.kernels.sm90.bwd.backward_config import (
            resolve_sm90_bwd_consumer_config,
        )
        from cudnn.flex_attention.kernels.sm100.bwd.backward_config import (
            resolve_sm100_bwd_consumer_config,
        )
        from cudnn.flex_attention.plan.kernels import normalize_arbitrary_block_sparse_config_bwd

        self.arch = 90 if torch.cuda.get_device_capability(query.device)[0] == 9 else 100
        self.varlen = query.ndim == 3
        self.heads = key.shape[-2]
        self.total_q = query.shape[-3] * 4
        folded = mask.for_physical_gqa(4)
        # Shape-only planner view. Its numerical interpretation is never read.
        view = query.view(*query.shape[:-3], self.total_q, self.heads, 128)
        self.plan = folded.plan_for(view, key, value, pack_gqa=False)
        packed, self.cuq, self.cuk = self.plan._runtime_args
        self.maxq = folded.max_seqlen_q if self.varlen else self.total_q
        self.batch = mask.cu_seqlens_q.numel() - 1 if self.varlen else query.shape[0]
        resolve_config = (
            resolve_sm90_bwd_consumer_config
            if self.arch == 90
            else resolve_sm100_bwd_consumer_config
        )
        self.maxk = folded.max_seqlen_k if self.varlen else key.shape[1]
        self.config = resolve_config(
            arch=self.arch,
            dtype=query.dtype,
            head_dim=128,
            head_dim_v=128,
            num_q_heads=self.heads,
            num_kv_heads=self.heads,
            is_varlen=self.varlen,
        )
        cfg = self.config
        self.tile_m = cfg.tile_m if self.arch == 90 else 128
        self.use_2cta = self.arch != 90 and cfg.cta_group_size == 2
        self.normalized = normalize_arbitrary_block_sparse_config_bwd(
            packed.bwd_tensors,
            device=query.device,
            batch_size=self.batch,
            num_q_heads=self.heads,
            is_varlen=self.varlen,
            block_size=cfg.block_size,
            subtile_factor=cfg.physical_subtiles,
            num_mma_threads=cfg.num_mma_threads,
            payload_padded_words=cfg.payload_padded_words,
            expected_hmask=1,
            expected_spt=cfg.spt,
            require_dq_write_order=True,
            expected_fixed_total_n_blocks=(
                None if self.varlen else self.batch * math.ceil(key.shape[1] / cfg.block_size[1])
            ),
        )
        self.sparse = _block_sparse_runtime_tuple(self.normalized)

    def backward(self, query, key, value, output, grad_output, lse, scale, grad_lse):
        import cutlass
        import cutlass.cute as cute
        from cudnn.flex_attention.plan.kernels import to_cute_block_sparse_tensors
        from cudnn.flex_attention.runtime.dsl_utils import to_cute_tensor
        from cutlass import Int32

        from aspec.kernels.cutedsl.cudnn_gqa.backward import FlexAttentionBackwardSm100
        from aspec.kernels.cutedsl.cudnn_gqa.backward_sm90 import FlexAttentionBackwardSm90
        from aspec.kernels.cutedsl.cudnn_gqa.postprocess import FlexAttentionBackwardPostprocess
        from aspec.kernels.cutedsl.cudnn_gqa.preprocess import FlexAttentionBackwardPreprocess

        grad_output = grad_output.contiguous()
        if grad_lse is not None:
            grad_lse = grad_lse.contiguous()
        # Match upstream per-sample padding, including ragged tails.
        padding_samples = self.batch + 1 if self.varlen else 1
        padded_q = ((self.total_q + padding_samples * self.tile_m - 1) // self.tile_m) * self.tile_m
        stat_shape = (self.heads, padded_q) if self.varlen else (self.batch, self.heads, padded_q)
        dpsum = torch.empty(stat_shape, device=query.device, dtype=torch.float32)
        log2_lse = torch.empty_like(dpsum)
        dq_accum = torch.empty(
            (*stat_shape[:-1], padded_q * 128), device=query.device, dtype=torch.float32
        )
        dq, dk, dv = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
        tensors = (
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            grad_lse,
            dpsum,
            log2_lse,
            dq_accum,
            dq,
            dk,
            dv,
            self.cuq,
            self.cuk,
        )
        cache_key = (
            tuple(_layout_key(t) for t in tensors),
            tuple(_layout_key(t, sparse=True) for t in self.sparse),
            self.arch,
            self.config,
            self.maxq,
            self.maxk,
        )
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(cache_key)
            if compiled is None:

                def tc(t):
                    return to_cute_tensor(t) if t is not None else None

                cq = to_cute_tensor(self.cuq, assumed_align=4) if self.varlen else None
                ck = to_cute_tensor(self.cuk, assumed_align=4) if self.varlen else None
                stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
                pre = cute.compile(
                    FlexAttentionBackwardPreprocess(cutlass.BFloat16, 128, 128, self.tile_m),
                    tc(output),
                    tc(grad_output),
                    tc(dpsum),
                    tc(lse),
                    tc(log2_lse),
                    tc(dq_accum),
                    cq,
                    tc(grad_lse),
                    Int32(self.maxq),
                    stream,
                    options="--enable-tvm-ffi",
                )
                cfg = self.config
                if self.arch == 90:
                    kernel = FlexAttentionBackwardSm90(
                        cutlass.BFloat16,
                        128,
                        128,
                        qhead_per_kvhead=1,
                        spt=cfg.spt,
                        tile_m=cfg.tile_m,
                        tile_n=cfg.tile_n,
                        Q_stage=cfg.num_stages_q,
                        dO_stage=cfg.num_stages_do,
                        PdS_stage=cfg.num_stages_pds,
                        SdP_swapAB=cfg.sdp_swap_ab,
                        dKV_swapAB=cfg.dkv_swap_ab,
                        AtomLayoutNdKV=cfg.atom_layout_n_dkv,
                        AtomLayoutMdQ=cfg.atom_layout_m_dq,
                        dQ_single_wg=cfg.dq_single_wg,
                    )
                else:
                    kernel = FlexAttentionBackwardSm100(
                        128,
                        128,
                        qhead_per_kvhead=1,
                        use_2cta_instrs=self.use_2cta,
                    )
                limits = (Int32(self.maxq), Int32(self.maxk)) if self.arch == 90 else ()
                main = cute.compile(
                    kernel,
                    tc(query),
                    tc(key),
                    tc(value),
                    tc(grad_output),
                    tc(log2_lse),
                    tc(dpsum),
                    tc(dq_accum),
                    tc(dk),
                    tc(dv),
                    scale,
                    cq,
                    ck,
                    None,
                    None,
                    None,
                    to_cute_block_sparse_tensors(self.normalized),
                    *limits,
                    stream,
                    options="--enable-tvm-ffi",
                )
                post = cute.compile(
                    FlexAttentionBackwardPostprocess(
                        cutlass.BFloat16,
                        128,
                        self.arch,
                        self.tile_m,
                        256 if self.arch == 90 and not cfg.dq_single_wg else 128,
                        AtomLayoutMdQ=cfg.atom_layout_m_dq if self.arch == 90 else 1,
                        use_2cta_instrs=self.use_2cta,
                        cluster_size=1,
                    ),
                    tc(dq_accum),
                    tc(dq),
                    scale,
                    cq,
                    Int32(self.maxq),
                    stream,
                    options="--enable-tvm-ffi",
                )
                compiled = (pre, main, post)
                _COMPILED[cache_key] = compiled
                if len(_COMPILED) > _MAX_COMPILED:
                    _COMPILED.popitem(last=False)
            _COMPILED.move_to_end(cache_key)
        pre, main, post = compiled
        pre(output, grad_output, dpsum, lse, log2_lse, dq_accum, self.cuq, grad_lse, self.maxq)
        main(
            query,
            key,
            value,
            grad_output,
            log2_lse,
            dpsum,
            dq_accum,
            dk,
            dv,
            scale,
            self.cuq,
            self.cuk,
            None,
            None,
            None,
            self.sparse,
            *((self.maxq, self.maxk) if self.arch == 90 else ()),
        )
        post(dq_accum, dq, scale, self.cuq, self.maxq)
        return dq, dk, dv


class _InternalGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale, pack_gqa):
        from cudnn.flex_attention import flex_attn_func

        validate_internal_gqa(query, key, value, mask)
        plan = mask.plan_for(query, key, value, pack_gqa=pack_gqa, build_backward=False)
        output, lse = flex_attn_func(
            query,
            key,
            value,
            mask_plan=plan,
            softmax_scale=scale,
            return_lse=True,
        )
        plan_key = mask._plan_key(query, key, value, False)
        backward_plan = mask._joint_backward_plans.get(plan_key)
        if backward_plan is None:
            backward_plan = _BackwardPlan(query, key, value, mask)
            mask._joint_backward_plans[plan_key] = backward_plan
        ctx.plan = backward_plan
        ctx.scale = scale
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query, key, value, output, lse)
        return output, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_lse):
        query, key, value, output, lse = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        dq, dk, dv = ctx.plan.backward(
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            ctx.scale,
            grad_lse,
        )
        return dq, dk, dv, None, None, None


def internal_gqa_attention(query, key, value, *, mask, scale, pack_gqa=None, return_lse=False):
    output, lse = _InternalGQA.apply(query, key, value, mask, scale, pack_gqa)
    return (output, lse) if return_lse else output
