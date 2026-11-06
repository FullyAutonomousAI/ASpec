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

"""Bounded content identities for external runtime code used by ASpec."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Patched files and their native v0.5.21 request/capture/runner dependencies.
# A bounded set catches divergent checkouts without hashing the entire repo.
SGLANG_RUNTIME_RELATIVE_PATHS = (
    "kernels/ops/attention/triton_gdn_fused_proj.py",
    "kernels/ops/attention/fla/chunk_delta_h.py",
    "kernels/ops/attention/fla/layernorm_gated.py",
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py",
    "kernels/ops/mamba/causal_conv1d_triton.py",
    "srt/entrypoints/engine.py",
    "srt/arg_groups/fields/exec_.py",
    "srt/arg_groups/fields/observability.py",
    "srt/layers/logits_processor.py",
    "srt/layers/aux_hidden_states.py",
    "srt/layers/layernorm_sp.py",
    "srt/managers/io_struct.py",
    "srt/managers/schedule_batch.py",
    "srt/managers/scheduler.py",
    "srt/managers/scheduler_components/batch_result_processor.py",
    "srt/managers/scheduler_components/output_streamer.py",
    "srt/managers/scheduler_components/flush_wrapper.py",
    "srt/managers/tokenizer_manager.py",
    "srt/managers/tp_worker.py",
    "srt/managers/utils.py",
    "srt/mem_cache/memory_pool.py",
    "srt/model_executor/forward_batch_info.py",
    "srt/model_executor/hook_manager.py",
    "srt/model_executor/model_runner.py",
    "srt/model_executor/runner/eager_runner.py",
    "srt/model_executor/model_runner_components/spec_aux_hidden_state.py",
    "srt/model_executor/model_runner_components/attention_backend_setup.py",
    "srt/model_executor/model_runner_components/cuda_graph_setup.py",
    "srt/models/qwen2.py",
    "srt/models/qwen3.py",
    "srt/models/qwen3_vl.py",
    "srt/models/qwen3_5.py",
    "srt/models/qwen3_next.py",
    "srt/models/qwen3_next_mtp.py",
    "srt/models/bailing_moe_v3.py",
    "srt/models/dflash.py",
    "srt/models/llama_eagle3.py",
    "srt/runtime_context.py",
    "srt/sampling/sampling_batch_info.py",
    "srt/server_args.py",
)


def module_runtime_identity(
    module: str,
    *,
    relative_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Hash an imported module or a bounded set of package-relative files."""

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        return {
            "path": None,
            "effective_tree_sha256": None,
            "file_count": 0,
            "missing_files": list(relative_paths or ()),
        }

    origin = Path(spec.origin).resolve()
    if relative_paths is None:
        identity_root = origin.parent
        files = [origin]
        missing: list[str] = []
    else:
        identity_root = origin.parent
        files = []
        missing = []
        for relative in relative_paths:
            candidate = identity_root / relative
            if candidate.is_file():
                files.append(candidate)
            else:
                missing.append(relative)

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(identity_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "path": str(origin),
        "effective_tree_sha256": digest.hexdigest() if files and not missing else None,
        "file_count": len(files),
        "missing_files": missing,
    }
