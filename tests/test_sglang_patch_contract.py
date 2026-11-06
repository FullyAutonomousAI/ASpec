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

from aspec.compat.sglang_patch import PATCH
from aspec.runtime_identity import SGLANG_RUNTIME_RELATIVE_PATHS


def test_runtime_identity_covers_every_file_in_fixed_sglang_patch():
    patch = PATCH.read_text(encoding="utf-8")
    changed_files = {
        line.removeprefix("diff --git a/python/sglang/").split(
            " b/python/sglang/", maxsplit=1
        )[0]
        for line in patch.splitlines()
        if line.startswith("diff --git a/python/sglang/")
    }

    assert changed_files <= set(SGLANG_RUNTIME_RELATIVE_PATHS)
    assert "srt/models/bailing_moe_v3.py" in SGLANG_RUNTIME_RELATIVE_PATHS
    assert "srt/managers/scheduler_components/batch_result_processor.py" in SGLANG_RUNTIME_RELATIVE_PATHS


def test_qwen3_base_model_streams_target_prefill_features_to_direct_sink():
    patch = PATCH.read_text(encoding="utf-8")
    qwen2_hunk = patch.split(
        "diff --git a/python/sglang/srt/models/qwen2.py", maxsplit=1
    )[1].split("diff --git ", maxsplit=1)[0]

    assert "direct_sink.capture(i, hidden_states, residual)" in qwen2_hunk
    assert "direct_sink.finish(hidden_states)" in qwen2_hunk
    assert "return hidden_states, direct_sink" in qwen2_hunk


def test_native_qwen3_and_qwen35_stream_capture_without_auxiliary_clones():
    patch = PATCH.read_text(encoding="utf-8")
    for filename in ("qwen3.py", "qwen3_5.py"):
        hunk = patch.split(
            f"diff --git a/python/sglang/srt/models/{filename}", maxsplit=1
        )[1].split("diff --git ", maxsplit=1)[0]
        assert "direct_sink.aux_accumulator" in hunk
        assert "direct_sink.finish(hidden_states)" in hunk
        assert "return hidden_states, direct_sink" in hunk


def test_scheduler_flush_reports_terminal_source_transport_status():
    patch = PATCH.read_text(encoding="utf-8")
    scheduler_hunk = patch.split(
        "diff --git a/python/sglang/srt/managers/scheduler.py", maxsplit=1
    )[1].split("diff --git ", maxsplit=1)[0]

    assert "publisher.drain()" in scheduler_hunk
    assert "return False" in scheduler_hunk
    assert "publisher.transport_status_message()" in patch
    assert "feature_publisher.resolve_outputs" in patch
    assert "spec_training_mooncake_store_keys" in patch


def test_hybrid_kernel_keeps_local_block_offsets_int32_and_global_addresses_int64():
    patch = PATCH.read_text()
    assert "+        T = (eos - bos).to(tl.int32)" in patch
    assert "i_t.to(tl.int64) * stride_h" in patch
    assert "i_n.to(tl.int64) * stride_track_state" in patch


def test_fused_qk_gate_widens_token_before_long_prefill_stride_multiplication():
    patch = PATCH.read_text()
    relative = "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py"
    hunk = patch.split(f"diff --git a/python/sglang/{relative}", maxsplit=1)[1]
    hunk = hunk.split("diff --git ", maxsplit=1)[0]
    assert "+    token = tl.program_id(0).to(tl.int64)" in hunk
    assert relative in SGLANG_RUNTIME_RELATIVE_PATHS
    # The real 197,474-token Qwen3.8-27B Q+gate input has a 12,288-element
    # stride. Casting the pointer after this multiplication is too late.
    stride = 2 * 24 * 256
    last_offset = (197474 - 1) * stride
    assert last_offset > 2**31 - 1
    assert ((last_offset + 2**31) % 2**32) - 2**31 < 0


def test_strided_gdn_gate_and_prefill_split_use_wide_global_offsets():
    patch = PATCH.read_text()
    gate = patch.split(
        "diff --git a/python/sglang/kernels/ops/attention/fla/layernorm_gated.py", 1
    )[1].split("diff --git ", 1)[0]
    assert "+    row_start = tl.program_id(0).to(tl.int64) * ROWS_PER_BLOCK" in gate
    assert "kernels/ops/attention/fla/layernorm_gated.py" in SGLANG_RUNTIME_RELATIVE_PATHS
    assert "+    i_t = tl.program_id(0).to(tl.int64)" in patch
    assert "+    token_offset = (BLOCK_M * chunk_offset).to(tl.int64)" in patch
    # Z may alias [q|k|v|z] instead of a contiguous head tensor. Even though
    # the flattened row index fits int32, token * physical stride does not.
    row = (197474 - 1) * 48 + 47
    assert row < 2**31
    assert (row // 48) * 16384 > 2**31 - 1


def test_dflash_moe_patch_reuses_official_components_and_fails_closed():
    patch = PATCH.read_text()
    dflash_hunk = patch.split(
        "diff --git a/python/sglang/srt/models/dflash.py", maxsplit=1
    )[1].split("diff --git ", maxsplit=1)[0]

    assert "Qwen3MoeSparseMoeBlock" in dflash_hunk
    assert "from sglang.srt.models.bailing_moe_v3 import BailingMoE" in dflash_hunk
    assert "FusedMoE.make_expert_params_mapping" in dflash_hunk
    assert "DFlash MoE checkpoint is missing required weights" in dflash_hunk
    assert "DFlash MoE serving has not validated expert parallelism" in dflash_hunk

    bailing_hunk = patch.split(
        "diff --git a/python/sglang/srt/models/bailing_moe_v3.py", maxsplit=1
    )[1].split("diff --git ", maxsplit=1)[0]
    assert 'getattr(config, "ffn_type", None) == "ling_moe"' in bailing_hunk
    assert 'getattr(config, "router_dtype", None) == "fp32"' in bailing_hunk


def test_exact_admitted_length_is_not_rejected_or_truncated():
    patch = PATCH.read_text()
    assert "+    if len(req.origin_input_ids) > max_req_input_len:" in patch
