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

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aspec.calibration import (
    CalibrationSearchSpace,
    CandidateMeasurement,
    build_calibration_report,
    candidate_identity_from_plan,
    compile_calibration_plan,
    load_candidate_measurement,
    select_candidate,
)
from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan

_ROOT = Path(__file__).parents[1]
_CONFIG = _ROOT / "configs" / "examples" / "sglang_qwen3_8b_dspark.yaml"


def _plan():
    overrides = [
        "training.draft_accumulation_steps=3",
        "training.num_train_steps=5",
    ]
    config = load_config(str(_CONFIG), cli_args=overrides)
    return compile_execution_plan(config, config_path=str(_CONFIG), cli_args=overrides)


def _measurement(candidate_id: str, throughput: float, resident: int):
    return CandidateMeasurement(
        candidate_id=candidate_id,
        evidence_path=f"/{candidate_id}/evidence.json",
        plan_id="sha256:plan",
        workload_id="sha256:workload",
        environment_id="sha256:environment",
        hardware_cohort_id="sha256:hardware",
        committed_work_per_second=throughput,
        optimizer_steps=3,
        peak_resident_bytes=resident,
        planned_sample_pool_size=8,
        completed=True,
        finite_training_signals=True,
        lifecycle_healthy=True,
        required_capabilities_active=True,
    )


def test_candidate_compiler_preserves_workload_and_rejects_invalid_splits():
    result = compile_calibration_plan(
        _plan(),
        total_gpus=8,
        search_space=CalibrationSearchSpace(
            trainer_gpu_counts=(2, 4, 6, 8),
            micro_batch_sizes=(1,),
            inference_batch_sizes=(2,),
            max_concurrent_batches=(1,),
            pool_headroom_steps=(0,),
        ),
    )

    assert result.rejection_counts == {"role_split_out_of_range": 1}
    assert [row.trainer_gpus for row in result.candidates] == [2, 4, 6]
    assert all(row.global_batch_size == 12 for row in result.candidates)
    assert [row.accumulation_steps for row in result.candidates] == [6, 3, 2]
    assert [row.sample_pool_size for row in result.candidates] == [16, 32, 48]
    assert all("training.max_concurrent_batches=1" in row.overrides() for row in result.candidates)


def test_packing_candidate_pool_cannot_fall_below_dispatch_source_minimum():
    overrides = [
        "training.dflash_packing=true",
        "training.packing_min_samples_per_dispatch=12",
        "training.micro_batch_size=1",
        "training.training_num_gpus_per_node=2",
        "training.draft_accumulation_steps=2",
        "training.prefetch_depth=0",
        "inference.inference_num_gpus=6",
        "inference.inference_num_gpus_per_node=6",
        "inference.inference_batch_size=1",
    ]
    config = load_config(str(_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(config, config_path=str(_CONFIG), cli_args=overrides)

    result = compile_calibration_plan(
        plan,
        total_gpus=8,
        search_space=CalibrationSearchSpace(
            trainer_gpu_counts=(2, 4),
            inference_batch_sizes=(1,),
        ),
    )

    assert [row.sample_pool_size for row in result.candidates] == [24, 12]


def test_candidate_compiler_preserves_fa4_cp_and_direct_transport_contracts():
    overrides = [
        "training.attention_backend=fa4",
        "training.draft_context_parallel_size=2",
        "training.training_num_gpus_per_node=2",
        "training.micro_batch_size=1",
        "training.draft_accumulation_steps=2",
        "training.max_concurrent_batches=2",
        "training.streaming_dispatch_with_training=true",
        "training.prefetch_depth=1",
        "inference.inference_num_gpus=6",
        "inference.inference_num_gpus_per_node=8",
        "inference.inference_batch_size=1",
        "inference.target_prefill_memory_optimized=true",
        "mooncake.data_plane=transfer_engine",
        "mooncake.protocol=nvlink_intra",
        "mooncake.require_nvlink=true",
        "mooncake.enable_gpu_direct=true",
        "mooncake.allow_host_fallback=false",
        "mooncake.source_gpu_ring_slots=4",
        "mooncake.destination_gpu_ring_slots=1",
    ]
    config = load_config(str(_CONFIG), cli_args=overrides)
    plan = compile_execution_plan(
        config,
        config_path=str(_CONFIG),
        cli_args=overrides,
    )

    result = compile_calibration_plan(
        plan,
        total_gpus=8,
        search_space=CalibrationSearchSpace(
            trainer_gpu_counts=(1, 2, 4, 6),
            max_concurrent_batches=(1, 2),
        ),
    )

    assert [row.trainer_gpus for row in result.candidates] == [2, 2, 4, 4]
    assert [row.accumulation_steps for row in result.candidates] == [2, 2, 1, 1]
    assert all(row.attention_backend == "fa4" for row in result.candidates)
    assert all(row.context_parallel_size == 2 for row in result.candidates)
    assert all(row.data_plane == "transfer_engine" for row in result.candidates)
    assert all(row.target_prefill_memory_optimized for row in result.candidates)
    assert result.rejection_counts == {
        "global_batch_not_divisible_by_dispatch": 1,
        "trainer_gpus_not_divisible_by_context_parallel": 1,
    }
    assert all(
        "training.draft_context_parallel_size=2" in row.overrides()
        for row in result.candidates
    )


def test_selector_gates_invalid_fast_run_and_prefers_capacity_at_parity():
    fastest_invalid = replace(_measurement("invalid", 20.0, 100), lifecycle_healthy=False)
    fastest = _measurement("fast", 10.0, 1000)
    lower_residency = _measurement("lean", 9.9, 500)

    result = select_candidate([fastest_invalid, fastest, lower_residency])

    assert result["recommended"]["candidate_id"] == "lean"
    assert result["rejected_measurements"] == 1


def test_selector_uses_planned_capacity_before_noisy_equivalent_throughput():
    larger = replace(
        _measurement("larger", 10.0, 500), planned_sample_pool_size=16
    )
    smaller = replace(
        _measurement("smaller", 9.9, 500), planned_sample_pool_size=8
    )

    result = select_candidate([larger, smaller])

    assert result["recommended"]["candidate_id"] == "smaller"


def _write_run(
    root: Path,
    *,
    target_gpus: int,
    hostname: str = "host",
    gpu_name: str = "NVIDIA H200",
    backend_path: str = "/example/sglang",
    hardware_label: str | None = None,
) -> Path:
    root.mkdir()
    plan = {
        "workload": {"models": {"target": "qwen"}, "duration": {"steps": 4}},
        "capabilities": [
            {"name": "target_features", "requirement": "required"},
            {"name": "performance_measurement", "requirement": "required"},
        ],
        "policy": {
            "parallelism": {
                "training": {
                    "attention_backend": "flex_attention",
                    "context_parallel": 1,
                }
            },
            "resources": {
                "training": {"world_size": 4},
                "inference": {"gpus": target_gpus, "gpus_per_engine": 1, "engines": target_gpus},
            },
            "pipeline": {
                "train_micro_batch_size": 1,
                "accumulation_steps": 2,
                "global_batch_size": 8,
                "request_batch_size": 2,
                "max_concurrent_batches": 1,
                "packed_rows": False,
            },
            "capacity": {
                "transfer_storage": {
                    "data_plane": "store",
                    "gpu_ring_slots": 2,
                    "source_gpu_ring_slots": None,
                    "destination_gpu_ring_slots": None,
                },
                "host_orchestrator": {"max_sample_pool_size": 8},
            },
        },
    }
    plan_payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    plan["plan_id"] = f"sha256:{hashlib.sha256(plan_payload).hexdigest()}"
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    evidence = {
        "plan_id": plan["plan_id"],
        "state": "completed",
        "health": "healthy",
        "source": {
            "aspec": {"source_tree_sha256": "source"},
            "backends": {
                "sglang": {
                    "path": backend_path,
                    "git_revision": "backend",
                    "git_dirty": False,
                }
            },
            "packages": {"torch": "1"},
            "host": {"hostname": hostname, "machine": "x86_64"},
        },
        "capabilities": [
            {
                "name": "target_features",
                "requirement": "required",
                "effective_state": "active",
            },
            {
                "name": "performance_measurement",
                "requirement": "required",
                "effective_state": "active",
            }
        ],
        "observations": {
            "runtime": {
                "accelerators": {
                    "state": "observed",
                    "visible_gpu_count": 8,
                    **(
                        {"hardware_label": hardware_label}
                        if hardware_label is not None
                        else {}
                    ),
                    "gpus": [
                        {
                            "index": index,
                            "reported_name": gpu_name,
                            "memory_mib": 143771,
                            "compute_capability": "9.0",
                            "driver_version": "570.1",
                        }
                        for index in range(8)
                    ],
                },
                "ray": {"cluster_resources": {"GPU": 8}},
            },
            "training": {
                "final_metrics": {"train/avg_loss": 1.0, "train/grad_norm": 2.0},
                "performance": {
                    "optimizer_steps": 2,
                    "committed_work_per_second": 4.0,
                },
            },
            "cleanup": {
                "state": "completed",
                "work_lifecycle": {
                    "state": "validated",
                    "resident_outputs": 0,
                    "resident_bytes": 0,
                    "peak_resident_bytes": 100,
                },
            },
        },
    }
    path = root / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_report_requires_one_workload_and_environment(tmp_path):
    first = _write_run(tmp_path / "first", target_gpus=4)
    second = _write_run(tmp_path / "second", target_gpus=2)

    report = build_calibration_report([first, second])

    assert report["valid_measurement_count"] == 2
    assert report["selection"]["recommended"]["candidate_id"] in {
        load_candidate_measurement(first).candidate_id,
        load_candidate_measurement(second).candidate_id,
    }

    same_hardware_other_host = _write_run(
        tmp_path / "third",
        target_gpus=4,
        hostname="other",
        gpu_name="Test GPU",
        backend_path="/another/sglang-checkout",
        hardware_label="NVIDIA H200",
    )
    cross_host_report = build_calibration_report([first, same_hardware_other_host])
    assert cross_host_report["valid_measurement_count"] == 2

    other_hardware = _write_run(
        tmp_path / "fourth",
        target_gpus=4,
        hostname="other",
        gpu_name="NVIDIA H100",
    )
    with pytest.raises(ValueError, match="one hardware cohort"):
        build_calibration_report([first, other_hardware])


def test_report_rejects_mismatched_plan_identity(tmp_path):
    evidence_path = _write_run(tmp_path / "run", target_gpus=4)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["plan_id"] = "sha256:other"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="identities do not match"):
        build_calibration_report([evidence_path])


def test_report_rejects_missing_performance_capability_evidence(tmp_path):
    evidence_path = _write_run(tmp_path / "run", target_gpus=4)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["capabilities"] = []
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="no valid candidate measurements"):
        build_calibration_report([evidence_path])


def test_report_rejects_any_missing_required_capability_evidence(tmp_path):
    evidence_path = _write_run(tmp_path / "run", target_gpus=4)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["capabilities"] = [
        item
        for item in evidence["capabilities"]
        if item["name"] != "target_features"
    ]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="no valid candidate measurements"):
        build_calibration_report([evidence_path])


def test_report_rejects_plan_content_that_breaks_declared_identity(tmp_path):
    evidence_path = _write_run(tmp_path / "run", target_gpus=4)
    plan_path = evidence_path.with_name("plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["policy"]["resources"]["training"]["world_size"] = 2
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="plan content does not match"):
        build_calibration_report([evidence_path])


def test_candidate_identity_covers_complete_semantics_but_not_output_dir():
    plan = _plan().to_dict()
    same_execution = json.loads(json.dumps(plan))
    same_execution["policy"]["artifacts"]["output_dir"] = "/another/repeat"
    changed_protocol = json.loads(json.dumps(plan))
    changed_protocol["policy"]["capacity"]["transfer_storage"]["protocol"] = "nvlink_intra"

    assert candidate_identity_from_plan(plan) == candidate_identity_from_plan(
        same_execution
    )
    assert candidate_identity_from_plan(plan) != candidate_identity_from_plan(
        changed_protocol
    )
