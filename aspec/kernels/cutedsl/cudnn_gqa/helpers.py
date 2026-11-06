# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# ASpec modifications Copyright (c) 2026 Ant Group Co., Ltd.
# Adapted from cuDNN Frontend 1.31.0 pack_gqa.py and runtime/dsl_utils.py.
# See README.md for provenance and the virtual row 4*t+r layout mapping.
import cutlass.cute as cute
from cudnn.flex_attention.runtime.dsl_utils import assume_tensor_aligned
from cutlass import Constexpr, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

GROUP = 4


@cute.jit
def fold_tokens(t, varlen: Constexpr):
    t = assume_tensor_aligned(t)
    if const_expr(varlen):
        shape = ((GROUP, t.shape[0]), t.shape[1] // GROUP, t.shape[2])
        stride = ((t.stride[1], t.stride[0]), t.stride[1] * GROUP, t.stride[2])
    else:
        shape = (t.shape[0], (GROUP, t.shape[1]), t.shape[2] // GROUP, t.shape[3])
        stride = (t.stride[0], (t.stride[2], t.stride[1]), t.stride[2] * GROUP, t.stride[3])
    return cute.make_tensor(t.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def fold_stats(t, varlen: Constexpr):
    if const_expr(varlen):
        shape = (t.shape[0] // GROUP, (GROUP, t.shape[1]))
        stride = (t.stride[0] * GROUP, (t.stride[0], t.stride[1]))
    else:
        shape = (t.shape[0], t.shape[1] // GROUP, (GROUP, t.shape[2]))
        stride = (t.stride[0], t.stride[1] * GROUP, (t.stride[1], t.stride[2]))
    return cute.make_tensor(t.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def token_sample(t, batch, head, offset, varlen: Constexpr):
    if const_expr(varlen):
        return cute.domain_offset(((0, offset // GROUP), 0), t[None, head, None])
    else:
        return t[batch, None, head, None]


@cute.jit
def stat_sample(t, batch, head, offset, varlen: Constexpr):
    if const_expr(varlen):
        return cute.domain_offset(((0, offset // GROUP),), t[None, head])
    else:
        return t[None, head, batch]



@dsl_user_op
def wait_cluster_relay(mbar_ptr, phase, *, loc=None, ip=None):
    """Acquire the cross-CTA relay; the DSL 4.7 wait helper is CTA-scoped."""
    llvm.inline_asm(
        None,
        [Int32(mbar_ptr.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
         Int32(phase).ir_value(loc=loc, ip=ip)],
        """{
            .reg .pred done;
        wait_relay:
            mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [$0], $1;
            @!done bra wait_relay;
        }""",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_cluster_relay(mbar_ptr, *, loc=None, ip=None):
    """Publish completed local dS shared memory to the leader CTA.

    The relay carries no GMEM payload. Keep cluster release ordering for its
    shared-memory data without the all-memory release that PTXAS lowers to a
    GPU-scoped memory drain. The leader retains its acquire.cluster wait.
    """
    llvm.inline_asm(
        None,
        [Int32(mbar_ptr.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b32 leader;
            mapa.shared::cluster.u32 leader, $0, 0;
            fence.release.sync_restrict::shared::cta.cluster;
            mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [leader];
        }""",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
