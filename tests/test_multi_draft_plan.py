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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from aspec.multi_draft import (
    compile_multi_draft_plan,
    experiment_contract,
    require_single_gpu_node,
)


def _config():
    root = Path(__file__).parents[1]
    base = OmegaConf.load(root / "configs/recipes/qwen3_8b_dspark.yaml")
    base.dataset.train_data_path = str(root / "examples/data/sample_conversations.jsonl")
    base.model.target_model_path = "Qwen/Qwen3-8B"
    config = load_config(base_config=base, save_snapshot=False)
    config.experiments = OmegaConf.create(
        [
            {"name": "a", "training": {"training_num_gpus_per_node": 1, "learning_rate": 0.0006}},
            {"name": "b", "training": {"training_num_gpus_per_node": 2, "learning_rate": 0.0003}},
        ]
    )
    return config


def test_plan_keeps_independent_training_and_one_target_allocation():
    config = _config()
    plan = compile_multi_draft_plan(config)
    assert plan.plan_id == compile_multi_draft_plan(config).plan_id
    assert plan.to_dict()["resources"] == {"target_gpus": 4, "draft_gpus": {"a": 1, "b": 2}}
    a, b = (p.to_runtime_args() for p in plan.plans.values())
    assert a.learning_rate != b.learning_rate
    assert a.global_batch_size == 128
    assert b.global_batch_size == 256
    assert a.output_dir != b.output_dir
    assert a.aspec_feature_contract == b.aspec_feature_contract


def test_source_seed_does_not_follow_first_experiment_rng():
    config = _config()
    config.training.seed = 123
    config.experiments[0].training.seed = 456
    plan = compile_multi_draft_plan(config)
    assert plan.to_runtime_args().seed == 123
    assert plan.plans["a"].to_runtime_args().seed == 456
    assert plan.to_dict()["shared_source"]["dataset_seed"] == 123


@pytest.mark.parametrize("logging", [
    {"use_tensorboard": True}, {"use_wandb": True}, {"report_to": "tensorboard"},
    {"report_to": "wandb"}, {"use_tensorboard": True, "use_wandb": True},
])
def test_shared_plan_accepts_tracking_switches(logging):
    config = _config()
    config.logging = OmegaConf.merge(config.logging, logging)
    plan = compile_multi_draft_plan(config)
    for name in ("a", "b"):
        args = plan.plans[name].to_runtime_args()
        assert args.use_tensorboard == (logging.get("use_tensorboard", False) or logging.get("report_to") == "tensorboard")
        assert args.use_wandb == (logging.get("use_wandb", False) or logging.get("report_to") == "wandb")
        assert plan.to_dict()["tracking"]["requested_backends"][name] == {
            "tensorboard": args.use_tensorboard, "wandb": args.use_wandb,
        }


def test_each_draft_may_override_tracking_without_changing_source():
    config = _config()
    config.experiments[0].logging = {"report_to": "tensorboard"}
    config.experiments[1].logging = {"report_to": "wandb"}
    plan = compile_multi_draft_plan(config)
    assert plan.plans["a"].to_runtime_args().use_tensorboard
    assert not plan.plans["a"].to_runtime_args().use_wandb
    assert plan.plans["b"].to_runtime_args().use_wandb
    assert not plan.plans["b"].to_runtime_args().use_tensorboard
    assert plan.to_dict()["resources"] == {"target_gpus": 4, "draft_gpus": {"a": 1, "b": 2}}


def test_unknown_tracking_backend_is_not_silently_ignored():
    config = _config()
    config.logging.report_to = "unknown"
    with pytest.raises(ValueError, match="logging.report_to"):
        compile_multi_draft_plan(config)


@pytest.mark.parametrize(
    "name",
    [
        "../a",
        "/tmp/a",
        "",
        "a/b",
        "a\\b",
        ".",
        "..",
        "config.yaml",
        "plan.json",
        "evidence.json",
        "a\n",
        "\x00",
    ],
)
def test_plan_rejects_unsafe_experiment_identity(name):
    config = _config()
    config.experiments[0].name = name
    with pytest.raises(ValueError, match="identity"):
        compile_multi_draft_plan(config)


@pytest.mark.parametrize("name", ["实验 A", "draft.lr=0.0006", "seed 42"])
def test_experiment_identity_allows_printable_hyperparameter_labels(name):
    config = _config()
    config.experiments[0].name = name
    plan = compile_multi_draft_plan(config)
    assert Path(plan.plans[name].to_runtime_args().output_dir).name == name


def test_plan_rejects_target_change_before_gpu_allocation():
    config = _config()
    config.experiments[0].model = {"target_model_path": "Qwen/Qwen3-4B"}
    with pytest.raises(ValueError, match="Target binding"):
        compile_multi_draft_plan(config)


def test_plan_rejects_duplicate_experiment_name():
    config = _config()
    config.experiments[1].name = "a"
    with pytest.raises(ValueError, match="duplicate"):
        compile_multi_draft_plan(config)


def test_plan_rejects_placement_override_instead_of_silently_ignoring_it():
    config = _config()
    config.experiments[1].training.placement_strategy = "training_first"
    with pytest.raises(ValueError, match="shared placement"):
        compile_multi_draft_plan(config)


def test_actual_multi_node_placement_is_rejected_before_starting_read_owners(monkeypatch):
    monkeypatch.setattr(
        "aspec.multi_draft.ray.util.placement_group_table",
        lambda _group: {"bundles_to_node_id": {0: "node-a", 1: "node-b"}},
    )
    with pytest.raises(NotImplementedError, match="physical node"):
        require_single_gpu_node({"training": ("pg", [0], [0]), "inference": ("pg", [1], [1])})


def test_experiment_feature_evidence_does_not_include_other_consumers():
    def participant(name, role, count):
        return {
            "participant": name,
            "role": role,
            "contract_id": "union" if role == "producer" else "view-" + name.split("/")[0],
            "metadata_validations": count,
            "tensor_validations": count,
            "transport": {
                "metrics": {
                    "destination_reads": count,
                    "destination_releases": count,
                    "destination_payload_bytes": 10 * count,
                    "destination_transfer_seconds": count,
                    "destination_max_transfer_seconds": count,
                }
            },
        }

    contract = {
        "contract_id": "union",
        "participants": [
            participant("target", "producer", 10),
            participant("a/trainer", "consumer", 10),
            participant("b/trainer", "consumer", 20),
        ],
        "consumer_count": 2,
        "transport": {},
        "data_admission": {
            "participants": [{"participant": "a/trainer"}, {"participant": "b/trainer"}]
        },
    }
    scoped = experiment_contract(contract, "a")
    assert scoped["source_contract_id"] == "union"
    assert scoped["contract_id"] == "view-a"
    assert scoped["consumer_count"] == 1
    assert scoped["consumer_metadata_validations"] == 10
    assert scoped["transport"]["destination_reads"] == 10
    assert scoped["producer_scope"] == "shared_session_not_experiment_work"
    assert len(contract["participants"]) == 3  # session evidence is not mutated

def test_shared_source_supports_independent_cp_and_packing_with_full_source_capacity():
    config = _config()
    config.training.attention_backend = "fa4"
    config.experiments[0].training.draft_context_parallel_size = 1
    config.experiments[1].training.draft_context_parallel_size = 2
    config.experiments[1].training.dflash_packing = True
    plan = compile_multi_draft_plan(config)
    assert plan.plans["a"].to_runtime_args().draft_context_parallel_size == 1
    assert plan.plans["b"].to_runtime_args().draft_context_parallel_size == 2
    assert plan.plans["b"].to_runtime_args().dflash_packing is True
    assert plan.to_runtime_args().draft_context_parallel_size == 1
    assert plan.source_plan is not None
    assert plan.to_dict()["shared_source"]["context_views"] == {"a": 1, "b": 2}


def test_shared_context_semantics_are_validated_before_gpu_allocation():
    config = _config()
    config.experiments[1].model = {"native_context_length": 65536}
    with pytest.raises(ValueError, match="source context"):
        compile_multi_draft_plan(config)


def test_full_source_does_not_force_first_draft_to_use_fa4(tmp_path):
    import json

    root = Path(__file__).parents[1]
    raw = json.loads((root / "aspec/config/dflash2_draft_config.json").read_text())
    raw["is_causal"] = True  # This existing DFlash2 combination requires Flex.
    draft = tmp_path / "causal.json"
    draft.write_text(json.dumps(raw))
    config = load_config(str(root / "configs/examples/qwen3_8b_dflash2.yaml"), save_snapshot=False)
    config.model.draft_model_config = str(draft)
    config.inference.store_last_hidden_states = True
    config.training.training_num_gpus_per_node = 1
    config.experiments = OmegaConf.create([
        {"name": "causal"},
        {"name": "cp2", "model": {"draft_model_config": str(
            root / "aspec/config/eagle3_qwen3_8b_draft_config.json"
        )}, "training": {"attention_backend": "fa4", "training_num_gpus_per_node": 2,
                        "draft_context_parallel_size": 2, "ttt_length": 3}},
    ])
    plan = compile_multi_draft_plan(config)
    assert plan.to_runtime_args().draft_context_parallel_size == 1
    assert plan.to_runtime_args().attention_backend == "flex_attention"
    assert plan.plans["cp2"].to_runtime_args().draft_context_parallel_size == 2


def test_source_admission_does_not_invent_first_experiments_dp_replicas():
    config = _config()
    config.training.attention_backend = "fa4"
    config.inference.inference_num_gpus = 1
    config.inference.inference_num_gpus_per_node = 1
    config.inference.inference_batch_size = 2
    config.mooncake.source_gpu_ring_slots = 2
    config.experiments[0].training.training_num_gpus_per_node = 4
    config.experiments[0].training.draft_context_parallel_size = 4
    config.experiments[1].training.training_num_gpus_per_node = 1
    config.experiments[1].training.draft_context_parallel_size = 1
    plan = compile_multi_draft_plan(config)
    assert plan.source_plan.to_runtime_args().dp_size == 1
    assert plan.to_dict()["resources"]["draft_gpus"] == {"a": 4, "b": 1}


@pytest.mark.parametrize("optimized", [False, True])
def test_common_cp_config_validates_the_single_physical_source_before_compilation(optimized):
    config = _config()
    config.training.attention_backend = "fa4"
    config.training.training_num_gpus_per_node = 4
    config.training.draft_context_parallel_size = 4
    config.inference.inference_num_gpus = 1
    config.inference.inference_num_gpus_per_node = 1
    config.inference.inference_batch_size = 2
    config.mooncake.source_gpu_ring_slots = 2
    config.inference.target_prefill_memory_optimized = optimized
    config.experiments[0].training.training_num_gpus_per_node = 4
    config.experiments[1].training.training_num_gpus_per_node = 1
    config.experiments[1].training.draft_context_parallel_size = 1
    loaded = load_config(base_config=config, save_snapshot=False)
    plan = compile_multi_draft_plan(loaded)
    assert plan.to_runtime_args().mooncake_source_gpu_ring_slots == 2
    assert plan.plans["a"].to_runtime_args().draft_context_parallel_size == 4
    loaded.experiments = []
    with pytest.raises(ValueError, match="source.*ring"):
        load_config(base_config=loaded, save_snapshot=False)
