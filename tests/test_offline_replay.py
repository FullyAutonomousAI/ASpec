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

"""Offline data and plan integration regression tests."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from aspec.config.train_config import load_config
from aspec.feature_contract import build_target_feature_contract
from aspec.offline.dataset import (
    OfflineDataset,
    bind_offline_dataset,
    target_identity,
)
from aspec.planning import compile_execution_plan


def binding(tmp_path, *, native_context_length=4096):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 32,
                "num_hidden_layers": 12,
                "max_position_embeddings": native_context_length,
            }
        )
    )
    (target / "model.safetensors").write_bytes(b"test-identity-only")
    contract = build_target_feature_contract(
        drafter_family="dspark",
        target_hidden_size=32,
        target_num_hidden_layers=12,
        aux_hidden_state_layers=[1, 9],
        store_last_hidden_states=True,
    )
    return target, {"target_identity": target_identity(target), "feature_contract": contract}


def record(contract, seq_len=8):
    return {
        "tensors": {
            "input_ids": torch.arange(seq_len),
            "hidden_states": torch.randn(seq_len, 64, dtype=torch.bfloat16),
            "last_hidden_states": torch.randn(seq_len, 32, dtype=torch.bfloat16),
        },
        "packed_loss_mask": [2, seq_len - 2],
        "metadata": {"aspec_feature_contract_id": contract["contract_id"]},
    }


def ready_dataset(tmp_path, *, native_context_length=4096):
    target, contract = binding(tmp_path, native_context_length=native_context_length)
    dataset = OfflineDataset(
        tmp_path / "data", create=True, binding=contract, last_hidden_states_prenorm=False
    )
    dataset.append("train", data_id="a", **record(contract["feature_contract"]))
    dataset.append("eval", data_id="b", **record(contract["feature_contract"]))
    dataset.finalize({"train": 1, "eval": 1})
    return target, OfflineDataset(dataset.root)


def test_offline_roundtrip_and_bound_target(tmp_path):
    target, dataset = ready_dataset(tmp_path)
    assert dataset.load("a")["hidden_states"].shape == (8, 64)
    args = Namespace(target_model_path=str(target), max_seq_length=16)
    bind_offline_dataset(dataset, args, dataset.metadata["binding"]["feature_contract"])
    assert args.last_hidden_states_prenorm is False
    assert args.offline_dataset_identity == dataset.identity
    assert dataset.count("eval") == 1


def test_partial_dataset_cannot_train_and_output_is_not_overwritten(tmp_path):
    _, contract = binding(tmp_path)
    dataset = OfflineDataset(
        tmp_path / "data", create=True, binding=contract, last_hidden_states_prenorm=False
    )
    with pytest.raises(ValueError, match="incomplete"):
        OfflineDataset(dataset.root)
    with pytest.raises(FileExistsError):
        OfflineDataset(dataset.root, create=True, binding=contract)
    with pytest.raises(ValueError, match="Incomplete"):
        dataset.finalize({"train": 1, "eval": 0})


@pytest.mark.parametrize(
    "field", ["hidden_states", "last_hidden_states", "input_ids", "packed_loss_mask"]
)
def test_corrupt_payload_rejected_before_writing_manifest(tmp_path, field):
    _, contract = binding(tmp_path)
    dataset = OfflineDataset(
        tmp_path / "data", create=True, binding=contract, last_hidden_states_prenorm=False
    )
    data = record(contract["feature_contract"])
    if field == "packed_loss_mask":
        data[field] = [2, 2]
    else:
        data["tensors"][field] = data["tensors"][field][:-1]
    with pytest.raises(ValueError):
        dataset.append("train", data_id="bad", **data)
    assert dataset.count("train") == 0
    assert not (dataset.root / "manifest.jsonl").exists()


def test_changed_sample_and_target_rejected(tmp_path):
    target, dataset = ready_dataset(tmp_path)
    row = dataset.rows("train")[0]
    (dataset.root / row["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        dataset.load("a")
    (target / "model.safetensors").write_bytes(b"other-target")
    with pytest.raises(ValueError, match="checkpoint identity"):
        bind_offline_dataset(
            dataset,
            Namespace(target_model_path=str(target), max_seq_length=16),
            dataset.metadata["binding"]["feature_contract"],
        )


def test_feature_layers_and_sequence_limit_are_checked(tmp_path):
    target, dataset = ready_dataset(tmp_path)
    args = Namespace(target_model_path=str(target), max_seq_length=4)
    contract = dataset.metadata["binding"]["feature_contract"]
    with pytest.raises(ValueError, match="max_seq_length"):
        bind_offline_dataset(dataset, args, contract)
    changed = copy.deepcopy(contract)
    changed["aux_hidden_state_layers"] = [2, 10]
    with pytest.raises(ValueError, match="feature contract"):
        bind_offline_dataset(dataset, args, changed)


def _offline_plan(tmp_path, *, max_seq_length=4096):
    target, dataset = ready_dataset(tmp_path, native_context_length=max_seq_length)
    root = Path(__file__).resolve().parents[1]
    draft = json.loads((root / "aspec/config/dspark_draft_config.json").read_text())
    draft.update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_target_layers=2,
        target_hidden_size=32,
        target_num_hidden_layers=12,
        target_layer_ids=[1, 9],
    )
    draft_file = tmp_path / "draft.json"
    draft_file.write_text(json.dumps(draft))
    config = load_config(str(root / "configs/examples/sglang_qwen3_8b_dspark.yaml"))
    config.model.target_model_path = str(target)
    config.model.native_context_length = max_seq_length
    config.model.draft_model_config = str(draft_file)
    config.training.training_num_gpus_per_node = 8
    config.training.max_seq_length = max_seq_length
    config.training.dspark_num_target_layers = 2
    config.inference.inference_engine_type = "offline"
    config.inference.offline.data_path = str(dataset.root)
    config.inference.inference_num_gpus = 0
    config.inference.target_prefill_memory_optimized = False
    config.mooncake = {
        "data_plane": "store",
        "protocol": "tcp",
        "enable_gpu_direct": False,
        "hidden_dim": 32,
        "host_buffer_size": "16MB",
        "global_segment_size": "32MB",
        "local_buffer_size": "16MB",
    }
    config.dataset.train_data_path = None
    config.dataset.eval_data_path = None
    config_file = tmp_path / "offline.yaml"
    OmegaConf.save(config, config_file)
    plan = compile_execution_plan(load_config(str(config_file), save_snapshot=False))
    return plan, dataset


def test_offline_plan_reports_cpu_replay_and_eight_draft_gpus(tmp_path):
    plan, dataset = _offline_plan(tmp_path)
    assert plan.policy.backends["target_features"] == "offline_replay"
    assert plan.policy.resources["training"]["world_size"] == 8
    assert plan.policy.resources["inference"]["gpus"] == 0
    assert plan.policy.resources["inference"]["gpus_per_engine"] == 0
    assert plan.policy.resources["inference"]["engines"] == 1
    assert plan.policy.capacity["host_orchestrator"]["inflight_sample_limit"] > 0
    assert plan.workload.data["offline_dataset_identity"] == dataset.identity
    assert plan.to_runtime_args().last_hidden_states_prenorm is False


def test_replay_records_do_not_retain_a_previous_execution_identity(tmp_path):
    from aspec.work_lifecycle import WORK_IDENTITY_KEYS, build_work_identity

    _, binding_value = binding(tmp_path)
    dataset = OfflineDataset(
        tmp_path / "data", create=True, binding=binding_value, last_hidden_states_prenorm=False
    )
    data = record(binding_value["feature_contract"])
    data["metadata"].update(build_work_identity(split="train", epoch=0, source_id="a"))
    dataset.append("train", data_id="a", **data)
    dataset.finalize({"train": 1, "eval": 0})
    assert not set(WORK_IDENTITY_KEYS).intersection(dataset.load("a")["metadata"])


@pytest.mark.parametrize("fail", [False, True])
def test_materialization_releases_controller_ownership_after_storage(fail, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aspec.offline import saving_actor
    from aspec.work_lifecycle import EXECUTION_ID_KEY

    cls = saving_actor.OfflineSavingActor.__ray_metadata__.modified_class
    saver = cls.__new__(cls)
    sample = SimpleNamespace(
        data_id="a",
        metadata={EXECUTION_ID_KEY: "execution"},
        mooncake_key="key",
        tensor_shapes={},
        tensor_dtypes={},
        packed_loss_mask=[2, 6],
    )
    saver.queues = {"train": Mock(get=Mock(return_value=sample))}
    saver.store, saver.dataset, saver.controller = Mock(), Mock(), Mock()
    saver.dataset.append.return_value = True
    if fail:
        saver.dataset.append.side_effect = OSError("write failed")
    monkeypatch.setattr(saving_actor.ray, "get", lambda value: value)
    if fail:
        with pytest.raises(OSError, match="write failed"):
            saver.save_from_queue("train")
    else:
        assert saver.save_from_queue("train") == 1
    saver.store.remove_eagle3_tensors.assert_called_once()
    saver.controller.release_inference_output.remote.assert_called_once_with(
        execution_id="execution",
        data_id="a",
        mooncake_key="key",
        participant="offline_saver",
        outcome="discarded_after_error" if fail else "consumed",
    )


def test_replay_actor_exposes_feature_validation_status():
    from unittest.mock import Mock

    from aspec.inference.engine.offline_replay_engine import OfflineReplayEngine

    engine = OfflineReplayEngine.__new__(OfflineReplayEngine)
    engine._validation = Mock()
    engine._validation.status.return_value = {"validated": 1}
    assert engine.get_feature_contract_status() == {"validated": 1}


def test_materialization_read_failure_releases_ownership(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aspec.offline import saving_actor
    from aspec.work_lifecycle import EXECUTION_ID_KEY

    cls = saving_actor.OfflineSavingActor.__ray_metadata__.modified_class
    saver = cls.__new__(cls)
    sample = SimpleNamespace(
        data_id="a",
        metadata={EXECUTION_ID_KEY: "execution"},
        mooncake_key="key",
        tensor_shapes={},
        tensor_dtypes={},
        packed_loss_mask=[2, 6],
    )
    saver.queues = {"train": Mock(get=Mock(return_value=sample))}
    saver.store, saver.dataset, saver.controller = Mock(), Mock(), Mock()
    saver.store.get.side_effect = OSError("read failed")
    monkeypatch.setattr(saving_actor.ray, "get", lambda value: value)
    with pytest.raises(OSError, match="read failed"):
        saver.save_from_queue("train")
    saver.dataset.append.assert_not_called()
    saver.store.remove_eagle3_tensors.assert_called_once()
    saver.controller.release_inference_output.remote.assert_called_once_with(
        execution_id="execution",
        data_id="a",
        mooncake_key="key",
        participant="offline_saver",
        outcome="discarded_after_error",
    )


def _replay_cleanup_fixture(monkeypatch, failure):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aspec.config.mooncake_config import MooncakeConfig
    from aspec.inference.engine import offline_replay_engine
    from aspec.transfer.mooncake.buffers import AsyncPutManager
    from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore

    engine = offline_replay_engine.OfflineReplayEngine.__new__(
        offline_replay_engine.OfflineReplayEngine
    )
    engine.args = SimpleNamespace(mooncake_feature_affinity="any")
    engine.rank = 0
    engine.get_node_ip = Mock(return_value="local")
    engine._rows_by_id = {"a": {}, "b": {}}
    engine._unpublished_keys = set()
    record = {
        "input_ids": torch.arange(4),
        "hidden_states": torch.zeros(4, 2, dtype=torch.bfloat16),
        "target": torch.zeros(4, 3, dtype=torch.bfloat16),
    }
    engine._dataset = Mock()
    engine._dataset.load.side_effect = (
        [record, ValueError("sample corrupt")] if failure == "sample" else [record]
    )
    engine._validation = Mock(contract_id="contract")
    monkeypatch.setattr(offline_replay_engine.uuid, "uuid4", lambda: "first")
    objects = set()
    raw = Mock()
    store = EagleMooncakeStore(MooncakeConfig(data_plane="store"))
    store._store = raw
    store._initialized = True
    store._init_event.set()
    manager = AsyncPutManager(store=raw, max_workers=1)
    store._async_put_manager = manager
    engine._mooncake_store = store

    def put(keys, tensors):
        # Completing the future represents a write arriving after the failure.
        future = Mock()
        future.result.side_effect = lambda: objects.update(keys)
        manager._in_flight[100] = future
        if failure == "put":
            raise ValueError("partial put failed")
    store._put_raw_tensors = put
    if failure == "flush":
        store.flush = Mock(side_effect=ValueError("flush failed"))

    def remove(keys, **kwargs):
        assert not manager._in_flight, "deletion raced an unfinished put"
        objects.difference_update(keys)
        return [0] * len(keys)
    raw.batch_remove.side_effect = remove
    return engine, store, manager, objects


@pytest.mark.parametrize("failure", ["sample", "put", "flush"])
def test_replay_failure_drains_writes_and_releases_all_unpublished_outputs(monkeypatch, failure):
    engine, store, manager, objects = _replay_cleanup_fixture(monkeypatch, failure)
    try:
        with pytest.raises(ValueError, match="corrupt|failed"):
            engine.generate(["a", "b"] if failure == "sample" else ["a"])
        assert not manager._in_flight
        assert not objects
        assert not engine._unpublished_keys
        store._store.batch_remove.assert_called_once()
        assert set(store._store.batch_remove.call_args.args[0]) == {
            "first_hs", "first_ids", "first_tgt",
        }
    finally:
        store.close()


def test_replay_retains_ownership_when_discard_fails(monkeypatch):
    from unittest.mock import Mock

    engine, store, manager, objects = _replay_cleanup_fixture(monkeypatch, "put")
    discard = store.discard_unpublished_objects
    store.discard_unpublished_objects = Mock(side_effect=RuntimeError("delete failed"))
    try:
        with pytest.raises(RuntimeError, match="delete failed"):
            engine.generate(["a"])
        assert engine._unpublished_keys == {"first_hs", "first_ids", "first_tgt"}
        assert not engine.health_check()
        store.discard_unpublished_objects = discard
        engine.shutdown()
        assert not objects
        assert not engine._unpublished_keys
        assert engine._mooncake_store is None
    finally:
        store.close()


def test_materialization_module_cli_is_importable():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "aspec.offline.generate", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--config" in result.stdout and "--output" in result.stdout


def test_replay_does_not_inherit_materializer_execution_metadata():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aspec.inference.engine.offline_replay_engine import OfflineReplayEngine

    engine = OfflineReplayEngine.__new__(OfflineReplayEngine)
    engine.args = SimpleNamespace(mooncake_feature_affinity="any")
    engine.rank = 2
    engine.get_node_ip = Mock(return_value="current-node")
    engine._rows_by_id = {"a": {"data_id": "a"}}
    engine._dataset = Mock()
    engine._dataset.load.return_value = {
        "input_ids": torch.arange(8),
        "hidden_states": torch.zeros(8, 4, dtype=torch.bfloat16),
        "packed_loss_mask": [2, 6],
        "metadata": {
            "source_node_ip": "old-node",
            "assigned_draft_node_ip": "old-node",
            "draft_attention_backend": "cudnn_flex_attention",
            "direct_cp_sharded": True,
            "request_id": "old-request",
        },
    }
    engine._validation = Mock(contract_id="current-contract")
    engine._mooncake_store = Mock()
    engine._mooncake_store.put.return_value = {"shapes": {}, "dtypes": {}}
    engine._unpublished_keys = set()
    output = engine._replay_one("a")
    metadata = output["metadata"]
    assert metadata["source_node_ip"] == "current-node"
    assert metadata["request_id"] == output["mooncake_key"]
    assert metadata["engine_rank"] == 2
    assert metadata["aspec_feature_contract_id"] == "current-contract"
    assert (
        not {"assigned_draft_node_ip", "draft_attention_backend", "direct_cp_sharded"}
        & metadata.keys()
    )


def test_materialization_does_not_inherit_training_or_profile_sample_budgets(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from aspec.controller.inference_manager import resolve_training_sample_budget
    from aspec.offline import generate

    args = SimpleNamespace(
        inference_engine_type="sgl",
        mooncake_data_plane="store",
        mooncake_protocol="tcp",
        draft_context_parallel_size=1,
        dflash_packing=False,
        num_train_steps=2,
        global_batch_size=2,
        topology_producer_profile_samples=4,
    )

    def stop_before_model_load(_):
        raise RuntimeError("checked mode before model load")

    monkeypatch.setattr(generate, "_get_draft_model_config", stop_before_model_load)
    with pytest.raises(RuntimeError, match="checked mode"):
        generate.materialize(args, str(tmp_path / "features"))
    assert (
        resolve_training_sample_budget(
            args.num_train_steps, args.global_batch_size, packing_enabled=False
        )
        is None
    )
    assert args.topology_producer_profile_samples == 0
    assert args.num_epochs == 1


def test_cpu_replay_host_buffer_works_without_cuda():
    from aspec.transfer.mooncake.buffers import HostBufferPool

    pool = HostBufferPool(buffer_size=256, pool_size=1, pin_memory=False)
    before = torch.cuda.is_initialized()
    pool.initialize()
    buffer = pool.get_buffer()
    source = torch.arange(16, dtype=torch.int64)
    assert buffer.copy_from_tensor(source) == source.numel() * source.element_size()
    assert not buffer._tensor.is_pinned()
    assert torch.equal(buffer._tensor[:128].view(torch.int64), source)
    assert torch.cuda.is_initialized() == before
    buffer.free()


def test_cpu_replay_factory_hides_cuda_devices(monkeypatch):
    from unittest.mock import Mock

    from aspec.inference import factory

    actor = Mock()
    remote = Mock(return_value=actor)
    monkeypatch.setattr(factory.ray, "remote", remote)
    factory._prepare_offline_replay_engines(Namespace(offline_num_engines=2), Mock())
    assert actor.options.call_count == 2
    for call in actor.options.call_args_list:
        assert call.kwargs["num_gpus"] == 0
        assert call.kwargs["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize(
    ("observed_length", "expected_state"),
    [(129000, "pending_runtime_evidence"), (131072, "active")],
)
def test_offline_long_context_evidence_requires_real_consumed_length(
    tmp_path, observed_length, expected_state
):
    from aspec.evidence import RunEvidenceRecorder

    plan, dataset = _offline_plan(tmp_path, max_seq_length=131072)
    recorder = RunEvidenceRecorder(plan, tmp_path / "evidence")
    recorder.record_actor_initialization(
        training_actor_count=8,
        inference_engine_count=1,
        inference_engine_statuses=[{
            "rank": 0,
            "engine_type": "offline",
            "num_gpus_per_engine": 0,
            "engine_gpu_ids": [],
            "dataset_identity": dataset.identity,
        }],
        dataset_size=1,
        eval_dataset_size=1,
        cluster_resources={"CPU": 16.0, "GPU": 8.0},
        alive_node_count=1,
    )
    assert recorder.data["observations"]["runtime"]["backend_tokens"]["state"] == "not_applicable_offline"
    recorder.record_cleanup({"work_lifecycle": {
        "state": "validated",
        "committed_work": plan.policy.pipeline["global_batch_size"],
        "produced_outputs": plan.policy.pipeline["global_batch_size"],
        "discarded_outputs": 0,
        "resident_outputs": 0,
        "resident_bytes": 0,
        "optimizer_steps_committed": 1,
        "optimizer_steps_contiguous": True,
    }})
    recorder.record_training({
        "optimizer_steps_observed": 1,
        "feature_contract": {
            "state": "validated",
            "contract_id": plan.workload.features["contract_id"],
            "producer_count": 1,
            "consumer_count": 8,
            "producer_metadata_validations": 8,
            "consumer_metadata_validations": 8,
            "consumer_tensor_validations": 8,
        },
        "scheduler": {
            "packing_enabled": False,
            "observed_sequence_max": observed_length,
            "observed_sequence_count": 8,
        },
    })
    capability = next(
        item for item in recorder.data["capabilities"]
        if item["name"] == "long_context_execution"
    )
    assert capability["effective_state"] == expected_state
    if expected_state == "active":
        observed = capability["evidence"][0]
        assert observed["observed_sequence_max"] == observed_length
        assert observed["backend_context_state"] == "not_applicable_offline"
