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

"""Real 128K native-asset proposal roundtrip; synthetic features, not a benchmark."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from aspec.draft_artifact import BlockDraftConsumer, MLAEagleConsumer, load_native_draft
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from tools.convert_to_hf import convert_checkpoint
from tools.validation.structured_thd_training import _config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", default=131072, type=int)
    parser.add_argument("--kind", choices=("all", "csa", "qsa", "dsa", "mla", "eagle_mla"), default="all")
    args = parser.parse_args()
    if args.length < 2:
        parser.error("--length must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for kind in (("csa", "qsa", "dsa", "mla", "eagle_mla") if args.kind == "all" else (args.kind,)):
        torch.manual_seed(741)
        if kind == "eagle_mla":
            raw = dict(architectures=["Eagle3DeepseekV2ForCausalLM"], hidden_size=32,
                       intermediate_size=64, vocab_size=64, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=2, target_hidden_size=32,
                       num_target_layers=2, num_aux_hidden_states=2, target_num_hidden_layers=6,
                       target_layer_ids=[1, 3], pad_token_id=0, pretraining_tp=1,
                       q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=128,
                       qk_rope_head_dim=64, v_head_dim=128)
            config = AutoDraftModelConfig.from_dict(raw)
        else:
            config = _config(kind)
            config.block_size = 2
            config.architectures = ["K3DSparkModel" if kind == "mla" else "Qwen3DSparkModel"]
        config.max_position_embeddings = args.length + 32
        model = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.bfloat16).cuda()
        if hasattr(model, "keep_csa_projection_precision"):
            model.keep_csa_projection_precision()
        checkpoint = args.output / kind / "iter_0000002"
        checkpoint.mkdir(parents=True)
        (checkpoint / "draft_config.json").write_text(json.dumps(config.to_dict()))
        dcp.save({"model_state": {"model": {"draft_model": model.state_dict()}}},
                 checkpoint_id=str(checkpoint / "model"))
        asset = args.output / kind / "asset"
        convert_checkpoint(input_dir=checkpoint, output_dir=asset, artifact_format="aspec")
        loaded = load_native_draft(asset, device="cuda")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
            assert value.dtype == loaded.state_dict()[key].dtype
        ids = torch.randint(0, 63, (1, args.length), device="cuda")
        features = torch.randn(1, args.length, 64, device="cuda", dtype=torch.bfloat16)
        if kind == "eagle_mla":
            expected = MLAEagleConsumer(model).propose(ids, features, num_tokens=4)
            actual = MLAEagleConsumer(loaded).propose(ids, features, num_tokens=4)
        else:
            # The public inference component must not run either training teacher.
            import aspec.models.draft.csa as csa
            import aspec.models.draft.dsa as dsa
            import aspec.models.draft.qsa as qsa
            def no_teacher(*_a, **_kw):
                raise AssertionError("Proposal computed a training-only indexer teacher")
            csa._selected_main_attention_target = no_teacher
            qsa._selected_teacher_target = no_teacher
            dsa._selected_teacher_target = no_teacher
            dsa._distributed_selected_teacher_target = no_teacher
            dsa._distributed_full_teacher_target = no_teacher
            head = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
            expected = BlockDraftConsumer(model, block_size=2).propose(ids, features, lm_head_weight=head)
            actual = BlockDraftConsumer(loaded, block_size=2).propose(ids, features, lm_head_weight=head)
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        records.append({"kind": kind, "length": args.length,
                        "all_tensors_exact": True, "mixed_dtypes_preserved": True,
                        "proposal_exact": True, "token_ids": actual["token_ids"].tolist()})
        print(records[-1], flush=True)
        del model, loaded, expected, actual, features
        torch.cuda.empty_cache()
    (args.output / "result.json").write_text(json.dumps({
        "scope": "synthetic feature/asset consumption correctness, not acceptance or speed",
        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
