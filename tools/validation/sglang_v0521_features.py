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

"""Real upstream Target -> strict NVLink features, with an HF semantic reference.

Run in an isolated venv with two explicitly selected, idle GPUs. This is a
feature/lifecycle check, not a throughput benchmark or drafter quality test.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from aspec.config.mooncake_config import MooncakeConfig
from aspec.runtime_identity import SGLANG_RUNTIME_RELATIVE_PATHS, module_runtime_identity
from aspec.transfer.mooncake import EagleMooncakeStore


def differences(actual, reference):
    delta = actual.float() - reference.float()
    denominator = reference.float().norm().clamp_min(1e-12)
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / denominator).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--layernorm-sp", action="store_true")
    parser.add_argument("--legacy-target", action="store_true", help="Validation only: run the pre-migration patched Target")
    parser.add_argument("--fixture", help="Save transported tensors for old/new or direct/buffered comparison")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()
    if args.tp < 1 or torch.cuda.device_count() != args.tp + 1:
        raise ValueError("Select TP+1 idle GPUs using CUDA_VISIBLE_DEVICES")
    if args.direct and args.tp != 1:
        raise ValueError("The existing direct-prefill capability requires Target TP1")
    if args.legacy_target and (args.direct or args.layernorm_sp):
        raise ValueError("Legacy comparison only covers the buffered baseline")
    receiver = torch.device("cuda", args.tp)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite validation output: {output}")
    fixture = Path(args.fixture) if args.fixture else None
    if fixture is not None and fixture.exists():
        raise ValueError(f"Refusing to overwrite feature fixture: {fixture}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config = getattr(config, "text_config", config)
    hidden = config.hidden_size
    layer_count = config.num_hidden_layers
    layers = [1, layer_count // 2 - 1, layer_count - 4]
    maximum = max(args.lengths)
    os.environ["ASPEC_RUN_ID"] = uuid.uuid4().hex
    os.environ["ASPEC_ATTEMPT_ID"] = uuid.uuid4().hex
    os.environ["ASPEC_ENGINE_RANK"] = "migration-feature-check"
    os.environ["SGLANG_SPEC_TRAINING_PACKAGE"] = "aspec"
    if args.direct:
        os.environ["ASPEC_TARGET_PREFILL_MEMORY_OPTIMIZED"] = "1"
    else:
        os.environ.pop("ASPEC_TARGET_PREFILL_MEMORY_OPTIMIZED", None)
    transport = MooncakeConfig(
        data_plane="transfer_engine", protocol="nvlink_intra",
        enable_gpu_direct=True, require_nvlink=True, allow_host_fallback=False,
        local_hostname="127.0.0.1", gpu_buffer_size="256MB",
        gpu_ring_slots=max(len(args.lengths), 2), destination_gpu_ring_slots=1,
        max_seq_len=maximum, hidden_dim=hidden, num_aux_layers=len(layers),
    )
    transport.export_env()
    namespace = SimpleNamespace(
        train_with_decode=False, aux_hidden_states_layers=layers,
        mooncake_hidden_dim=hidden, draft_context_parallel_size=1,
        max_seq_length=maximum, store_last_hidden_states=True,
        target_prefill_memory_optimized=args.direct,
        attention_backend="cudnn_flex_attention",
    )
    kwargs = {
        "model_path": args.model, "tp_size": args.tp, "pp_size": 1,
        "aux_hidden_state_layer_ids": layers, "base_gpu_id": 0,
        "mem_fraction_static": 0.6, "max_total_tokens": sum(args.lengths) + 1024,
        "disable_radix_cache": True, "disable_cuda_graph": True,
        "enable_return_hidden_states": True, "chunked_prefill_size": -1,
        "log_level": "warning", "trust_remote_code": True,
    }
    import sglang as sgl

    if args.legacy_target:
        kwargs.update(enable_aux_hidden_states=True, enable_spec_training_mooncake=True,
                      spec_training_store_last_hidden_states=True)
        engine = sgl.Engine(**kwargs)
    else:
        from aspec.inference.sglang_v0521 import TrainingEngine, training_engine_kwargs

        kwargs["enable_layernorm_sp"] = args.layernorm_sp
        engine = TrainingEngine(sgl.Engine(**training_engine_kwargs(kwargs, namespace)))
    reader = EagleMooncakeStore(transport)
    observations = []
    try:
        reader.setup(device=receiver, p2p_role="destination")
        input_ids = [list(range(32, 32 + length)) for length in args.lengths]
        response = engine.generate(
            input_ids=input_ids, rid=[f"r{i}" for i in range(len(input_ids))],
            spec_training_data_id=[f"w{i}" for i in range(len(input_ids))],
            packed_loss_mask=None, sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
        )
        captured = []
        for tokens, item in zip(input_ids, response, strict=True):
            keys = item["meta_info"]["spec_training_mooncake_store_keys"]
            if len(keys) != 1 or not isinstance(keys[0], str):
                raise RuntimeError(f"Native IPC descriptor count={len(keys)}, element types="
                                   f"{[type(key).__name__ for key in keys[:3]]}")
            length = len(tokens)
            shapes = {
                "hidden_states": (length, hidden * len(layers)),
                "input_ids": (length,), "last_hidden_states": (length, hidden),
            }
            tensors = reader.get(
                keys[0], shapes=shapes,
                dtypes={"hidden_states": torch.bfloat16, "last_hidden_states": torch.bfloat16,
                        "input_ids": torch.int64}, device=receiver,
            )
            try:
                if not torch.equal(tensors.input_ids.cpu(), torch.tensor(tokens)):
                    raise RuntimeError("Transport changed input token identity")
                captured.append((tensors.hidden_states.cpu(), tensors.last_hidden_states.cpu()))
            finally:
                tensors.p2p_receive_lease.release()
        result = engine.loop.run_until_complete(engine.tokenizer_manager.flush_cache(timeout_s=30))
        if not result.success:
            raise RuntimeError(f"Source did not drain: {result.message}")
        source_status = json.loads(result.message) if result.message else None
        if fixture is not None:
            fixture.mkdir(parents=True)
            torch.save({"input_ids": input_ids, "layers": layers, "captured": captured}, fixture / "features.pt")
        reference = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="eager",
        ).to(receiver).eval()
        with torch.no_grad():
            for tokens, (aux, teacher) in zip(input_ids, captured, strict=True):
                values = reference(
                    torch.tensor([tokens], device=receiver),
                    output_hidden_states=True, use_cache=False,
                )
                expected = torch.cat([values.hidden_states[i + 1][0] for i in layers], -1).cpu()
                observations.append({
                    "length": len(tokens), "aux": differences(aux, expected),
                    "teacher": differences(teacher, values.hidden_states[-1][0].cpu()),
                })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "sglang_path": sgl.__file__, "direct": args.direct,
            "tp": args.tp, "layernorm_sp": args.layernorm_sp,
            "legacy_target": args.legacy_target, "fixture": str(fixture) if fixture else None,
            "runtime_identity": module_runtime_identity("sglang", relative_paths=SGLANG_RUNTIME_RELATIVE_PATHS),
            "dependencies": {name: version(name) for name in (
                "sglang", "torch", "transformers", "flash-attn-4", "apache-tvm-ffi",
                "mooncake-transfer-engine",
            )},
            "layers": layers, "observations": observations,
            "source_transport": source_status, "destination_transport": reader.get_p2p_status(),
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(observations, indent=2), flush=True)
    finally:
        reader.close()
        engine.shutdown()


if __name__ == "__main__":
    main()
