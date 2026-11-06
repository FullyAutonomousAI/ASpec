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

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from aspec import topology_runner
from aspec.calibration import CandidateMeasurement
from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan
from aspec.topology import (
    CapabilityBundle,
    ServiceObservation,
    TopologyPolicy,
    compile_topology_study,
    current_capability_bundle,
    rank_topology_candidates,
)
from aspec.topology_runner import AutomaticRunPolicy

_ROOT = Path(__file__).parents[1]
_CONFIG = _ROOT / "configs" / "examples" / "sglang_qwen3_8b_dspark.yaml"


def _study():
    config = load_config(str(_CONFIG))
    plan = compile_execution_plan(config, config_path=str(_CONFIG))
    return compile_topology_study(
        config,
        plan,
        total_gpus=8,
        policy=TopologyPolicy(
            buffer_headroom_dispatches=(0, 1),
            profile_headroom_dispatches=1,
            max_finalist_shapes=2,
        ),
        config_path=str(_CONFIG),
    )


def test_topology_study_derives_role_splits_and_recompiles_canonical_plans():
    study = _study()

    assert {row.trainer_gpus for row in study.candidates} == {1, 2, 4}
    assert {row.target_tensor_parallel_size for row in study.candidates} == {1, 2, 4}
    assert len(study.candidates) == 12
    profile_candidates = [study.candidate(value) for value in study.profile_candidate_ids]
    assert [row.target_tensor_parallel_size for row in profile_candidates] == [
        1,
        1,
        2,
        2,
        4,
    ]
    assert [row.trainer_gpus for row in profile_candidates] == [1, 4, 2, 4, 4]
    assert all(row.global_batch_size == 8 for row in study.candidates)
    assert all(row.plan["workload"] == study.candidates[0].plan["workload"] for row in study.candidates)
    assert all(row.plan_id == row.plan["plan_id"] for row in study.candidates)


def test_explicit_target_tp_ladder_preserves_expert_order():
    config = load_config(str(_CONFIG))
    plan = compile_execution_plan(config, config_path=str(_CONFIG))

    bundle = current_capability_bundle(
        plan, target_tensor_parallel_sizes=(4, 2, 1, 2)
    )
    study = compile_topology_study(
        config,
        plan,
        total_gpus=8,
        bundles=(bundle,),
        config_path=str(_CONFIG),
    )

    assert bundle.target_tensor_parallel_sizes == (4, 2, 1)
    profile_tps = [
        study.candidate(candidate_id).target_tensor_parallel_size
        for candidate_id in study.profile_candidate_ids
    ]
    assert profile_tps == [4, 2, 2, 1, 1]


def test_tcp_bundle_preserves_registered_cuda_without_claiming_network_gpu_direct():
    config = load_config(str(_CONFIG))
    config.training.streaming_dispatch_with_training = True
    config.mooncake.data_plane = "transfer_engine"
    config.mooncake.protocol = "tcp"
    config.mooncake.enable_gpu_direct = True
    config.mooncake.allow_host_fallback = False
    config.mooncake.require_nvlink = False
    plan = compile_execution_plan(config)

    bundle = current_capability_bundle(plan)

    assert bundle.gpu_direct is True
    gpu_direct_claim = next(
        item for item in plan.capabilities if item.name == "gpu_direct_transport"
    )
    assert gpu_direct_claim.effective_state == "omitted"
    study = compile_topology_study(config, plan, total_gpus=8, bundles=(bundle,))
    assert study.candidates


def test_service_rate_model_selects_balanced_shapes_and_keeps_buffer_variants():
    study = _study()
    observations = []
    for candidate_id in study.profile_candidate_ids:
        candidate = study.candidate(candidate_id)
        producer = 10.0 * candidate.target_engines
        consumer = 8.0 * candidate.logical_data_parallel_size
        observations.append(
            ServiceObservation(
                session_id=study.session_id,
                candidate_id=candidate.candidate_id,
                evidence_path=f"/{candidate.label}/evidence.json",
                environment_id="sha256:environment",
                bundle_id=candidate.bundle_id,
                target_tensor_parallel_size=candidate.target_tensor_parallel_size,
                target_engines=candidate.target_engines,
                inference_batch_size=candidate.inference_batch_size,
                max_concurrent_batches=candidate.max_concurrent_batches,
                context_parallel_size=candidate.context_parallel_size,
                logical_data_parallel_size=candidate.logical_data_parallel_size,
                micro_batch_size=candidate.micro_batch_size,
                producer_work_per_second=producer,
                consumer_work_per_second=consumer,
                end_to_end_work_per_second=0.9 * min(producer, consumer),
                committed_work_per_training_row=1.0,
            )
        )

    result = rank_topology_candidates(study, observations)
    finalists = [study.candidate(value) for value in result["finalist_candidate_ids"]]

    assert result["ranked_candidates"][0]["predicted_end_to_end_work_per_second"] == 28.8
    assert {row.trainer_gpus for row in finalists} == {4}
    assert {row.target_tensor_parallel_size for row in finalists} == {1, 2}
    assert {row.pool_headroom_dispatches for row in finalists} == {0, 1}
    assert all(row.session_id == study.session_id for row in observations)
    assert asdict(study.policy)["max_finalist_shapes"] == 2


def test_runner_waits_for_launched_gpu_contexts_to_quiesce(monkeypatch):
    observations = iter(
        [
            {"ray": [], "gpu": ["123, [No data]"]},
            {"ray": [], "gpu": ["123, [No data]"]},
            {"ray": [], "gpu": []},
        ]
    )
    sleeps = []
    monkeypatch.setattr(
        topology_runner, "_active_runtime_processes", lambda: next(observations)
    )
    monkeypatch.setattr(topology_runner.time, "sleep", sleeps.append)

    clean, active, elapsed = topology_runner._wait_for_runtime_quiescence(
        timeout_seconds=10.0,
        poll_seconds=0.01,
    )

    assert clean is True
    assert active == {"ray": [], "gpu": []}
    assert elapsed >= 0
    assert sleeps == [0.01, 0.01]


def test_runner_quiescence_timeout_must_be_positive():
    with pytest.raises(ValueError, match="post_run_quiescence_timeout_seconds"):
        AutomaticRunPolicy(post_run_quiescence_timeout_seconds=0)


def test_runner_invalidates_completed_attempt_when_runtime_does_not_quiesce(
    monkeypatch,
):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.run_policy = AutomaticRunPolicy(
        post_run_quiescence_timeout_seconds=3.0
    )
    attempt = {
        "state": "completed",
        "duration_seconds": 10.0,
        "error": None,
    }
    active = {"ray": ["123 raylet"], "gpu": ["123, python"]}
    monkeypatch.setattr(
        topology_runner,
        "_wait_for_runtime_quiescence",
        lambda **_: (False, active, 3.0),
    )

    runner._record_post_run_quiescence(attempt)

    assert attempt["state"] == "invalid"
    assert attempt["duration_seconds"] == 13.0
    assert attempt["post_run_quiescence_state"] == "timed_out"
    assert "did not release Ray/GPU resources" in attempt["error"]
    assert "123 raylet" in attempt["error"]


def test_runner_retries_orphaned_attempt_without_terminal_evidence(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "attempt-01"
    run_dir.mkdir()
    (run_dir / "train.log").write_text("started\n", encoding="utf-8")
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner._state = {
        "attempts": [
            {
                "candidate_id": "candidate-a",
                "phase": "winner_confirmation",
                "state": "running",
                "run_dir": str(run_dir),
                "started_at_unix": 1.0,
                "duration_seconds": None,
                "error": None,
            }
        ]
    }
    saves = []
    runner._save = lambda: saves.append(True)
    monkeypatch.setattr(
        topology_runner,
        "_active_runtime_processes",
        lambda: {"ray": [], "gpu": []},
    )

    runner._recover_orphaned_attempts()

    attempt = runner._state["attempts"][0]
    assert attempt["state"] == "interrupted"
    assert attempt["post_run_quiescence_state"] == "clean_on_resume"
    assert attempt["duration_seconds"] > 0
    assert runner._prior_run_seconds == attempt["duration_seconds"]
    assert runner._should_run_phase("candidate-a", "winner_confirmation")
    assert saves == [True]


def test_finalist_selection_ignores_interrupted_confirmation(monkeypatch):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.run_policy = AutomaticRunPolicy()
    runner._state = {
        "attempts": [
            {
                "candidate_id": "candidate-a",
                "phase": "short_e2e",
                "state": "completed",
            },
            {
                "candidate_id": "candidate-a",
                "phase": "winner_confirmation",
                "state": "interrupted",
            },
            {
                "candidate_id": "candidate-b",
                "phase": "role_profile",
                "state": "completed",
            },
        ]
    }
    measurements = [
        SimpleNamespace(
            environment_id="environment",
            hardware_cohort_id="hardware",
            workload_id="workload",
        )
    ]
    runner._completed_measurements = lambda *args, **kwargs: measurements
    monkeypatch.setattr(
        topology_runner,
        "select_candidate",
        lambda rows, **kwargs: {
            "environment_id": rows[0].environment_id,
            "recommended": {"candidate_id": "candidate-a"},
        },
    )

    selection = runner._select_finalist({"candidate-a", "candidate-b"})

    assert selection["eligible_candidate_ids"] == ["candidate-a", "candidate-b"]
    assert selection["ineligible_candidates_with_failed_attempts"] == []
    assert not runner._should_run_selection_candidate("candidate-b")


def test_failed_selection_attempt_is_not_silently_retried():
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner._state = {
        "attempts": [
            {
                "candidate_id": "candidate-a",
                "phase": "short_e2e",
                "state": "failed",
            }
        ]
    }

    assert not runner._should_run_selection_candidate("candidate-a")


def test_role_profile_failure_only_removes_affected_profile(monkeypatch):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.run_policy = AutomaticRunPolicy(confirm_repeats=0)
    bundle = SimpleNamespace(
        bundle_id="bundle", name="bundle", target_tensor_parallel_sizes=(1, 2)
    )
    candidates = {
        candidate_id: SimpleNamespace(
            candidate_id=candidate_id,
            bundle_id=bundle.bundle_id,
            target_tensor_parallel_size=target_tp,
        )
        for candidate_id, target_tp in (("failed", 1), ("healthy", 1), ("tp2", 2))
    }
    runner.study = SimpleNamespace(
        bundles=(bundle,),
        profile_candidate_ids=tuple(candidates),
        candidate=candidates.__getitem__,
    )
    runner._state = {
        "state": "running",
        "attempts": [
            {
                "candidate_id": "failed",
                "phase": "role_profile",
                "state": "failed",
                "error": "out of memory",
            },
            {
                "candidate_id": "healthy",
                "phase": "role_profile",
                "state": "completed",
                "evidence_path": "/healthy/evidence.json",
            },
        ],
    }
    runner._recover_orphaned_attempts = lambda: None
    runner._should_run_phase = lambda *_: False
    runner._should_run_selection_candidate = lambda *_: False
    runner._save = lambda: None
    runner._select_finalist = lambda _: {
        "recommended": {"candidate_id": "healthy"},
        "ranked_equivalent": [{"candidate_id": "healthy"}],
    }
    observation = SimpleNamespace(candidate_id="healthy")
    monkeypatch.setattr(
        topology_runner, "load_service_observation", lambda *_: observation
    )
    monkeypatch.setattr(
        topology_runner,
        "rank_topology_candidates",
        lambda _study, rows: {
            "finalist_candidate_ids": ["healthy"],
            "observed_candidate_ids": [row.candidate_id for row in rows],
        },
    )

    state = runner._run_session()

    assert state["state"] == "completed"
    assert state["ranking"]["observed_candidate_ids"] == ["healthy"]
    assert state["ranking"]["failed_role_profiles"][0]["candidate_id"] == "failed"
    assert state["ranking"]["failed_role_profiles"][0]["states"] == ["failed"]
    assert state["ranking"]["failed_role_profiles"][0]["errors"] == [
        "out of memory"
    ]
    assert state["target_tp_resolution"][0][
        "selected_target_tensor_parallel_size"
    ] == 1
    assert state["target_tp_resolution"][0]["tiers"][1]["state"] == (
        "skipped_after_lower_tp_selected"
    )


def _tp_ladder_runner():
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    bundle = SimpleNamespace(
        bundle_id="bundle", name="bundle", target_tensor_parallel_sizes=(1, 2, 4)
    )
    candidates = {
        f"tp{target_tp}": SimpleNamespace(
            candidate_id=f"tp{target_tp}",
            bundle_id=bundle.bundle_id,
            target_tensor_parallel_size=target_tp,
        )
        for target_tp in bundle.target_tensor_parallel_sizes
    }
    runner.study = SimpleNamespace(
        bundles=(bundle,),
        profile_candidate_ids=tuple(candidates),
        candidate=candidates.__getitem__,
    )
    runner._state = {"attempts": []}
    runner._save = lambda: None
    return runner


def test_target_tp_ladder_stops_after_first_healthy_tier(monkeypatch):
    runner = _tp_ladder_runner()
    launched = []

    def run_candidate(candidate, **_):
        launched.append(candidate.candidate_id)
        attempt = {
            "candidate_id": candidate.candidate_id,
            "phase": "role_profile",
            "state": "completed",
            "evidence_path": f"/{candidate.candidate_id}/evidence.json",
        }
        runner._state["attempts"].append(attempt)
        return attempt

    runner._run_candidate = run_candidate
    monkeypatch.setattr(
        topology_runner,
        "load_service_observation",
        lambda _study, path: SimpleNamespace(candidate_id=Path(path).parent.name),
    )

    observations, failures = runner._resolve_target_tp_profiles()

    assert launched == ["tp1"]
    assert [row.candidate_id for row in observations] == ["tp1"]
    assert failures == []
    assert [
        tier["state"] for tier in runner._state["target_tp_resolution"][0]["tiers"]
    ] == [
        "selected",
        "skipped_after_lower_tp_selected",
        "skipped_after_lower_tp_selected",
    ]


def test_target_tp_ladder_advances_only_after_target_capacity_failure(monkeypatch):
    runner = _tp_ladder_runner()
    launched = []

    def run_candidate(candidate, **_):
        launched.append(candidate.candidate_id)
        attempt = {
            "candidate_id": candidate.candidate_id,
            "phase": "role_profile",
            "state": "completed" if candidate.candidate_id == "tp2" else "failed",
            "error": None,
        }
        if attempt["state"] == "completed":
            attempt["evidence_path"] = f"/{candidate.candidate_id}/evidence.json"
        else:
            attempt["error"] = "target capacity"
            attempt["failure_classification"] = {
                "kind": "target_capacity_infeasible",
                "signals": ["profiled_token_capacity"],
                "source": "evidence.error",
            }
        runner._state["attempts"].append(attempt)
        return attempt

    runner._run_candidate = run_candidate
    monkeypatch.setattr(
        topology_runner,
        "load_service_observation",
        lambda _study, path: SimpleNamespace(candidate_id=Path(path).parent.name),
    )

    observations, _ = runner._resolve_target_tp_profiles()

    assert launched == ["tp1", "tp2"]
    assert [row.candidate_id for row in observations] == ["tp2"]
    result = runner._state["target_tp_resolution"][0]
    assert result["selected_target_tensor_parallel_size"] == 2
    assert [tier["state"] for tier in result["tiers"]] == [
        "target_capacity_fallback",
        "selected",
        "skipped_after_lower_tp_selected",
    ]


def test_target_tp_ladder_refuses_generic_failure(monkeypatch):
    runner = _tp_ladder_runner()
    launched = []

    def run_candidate(candidate, **_):
        launched.append(candidate.candidate_id)
        attempt = {
            "candidate_id": candidate.candidate_id,
            "phase": "role_profile",
            "state": "failed",
            "error": "trainer failed",
            "failure_classification": {
                "kind": "unclassified_failure",
                "signals": [],
                "source": None,
            },
        }
        runner._state["attempts"].append(attempt)
        return attempt

    runner._run_candidate = run_candidate

    with pytest.raises(ValueError, match="refusing to increase TP"):
        runner._resolve_target_tp_profiles()

    assert launched == ["tp1"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "ray::SglEngine.init()\nRuntimeError: SGLang profiled "
            "max_total_tokens cannot satisfy the requested Target-prefill concurrency",
            "target_capacity_infeasible",
        ),
        (
            "ray::TrainerActor.init()\ntorch.OutOfMemoryError: CUDA out of memory",
            "unclassified_failure",
        ),
        (
            "class_name: SglEngine\nSome common causes include the process being "
            "killed by the OOM killer due to high memory usage",
            "unclassified_failure",
        ),
        (
            "ray::SglEngine.init()\n"
            + ("target initialized\n" * 400)
            + "ray::TrainerActor.train()\ntorch.OutOfMemoryError: CUDA out of memory",
            "unclassified_failure",
        ),
    ],
)
def test_role_profile_failure_classifier_requires_target_capacity_evidence(
    tmp_path, message, expected
):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        '{"error": {"type": "RuntimeError", "message": '
        + repr(message).replace("'", '"')
        + "}}",
        encoding="utf-8",
    )
    attempt = {"evidence_path": str(evidence_path)}

    result = topology_runner._classify_role_profile_failure(attempt)

    assert result["kind"] == expected


def _hardware(*, link: str = "NV18", second_name: str = "NVIDIA H200"):
    gpu_count = 2
    rows = ["\tGPU0\tGPU1"]
    rows.append(f"GPU0\tX\t{link}")
    rows.append(f"GPU1\t{link}\tX")
    return {
        "hardware_label": "H200",
        "visible_gpu_count": gpu_count,
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-a",
                "reported_name": "NVIDIA H200",
                "memory_mib": 143771,
                "compute_capability": "9.0",
                "driver_version": "570.1",
            },
            {
                "index": 1,
                "uuid": "GPU-b",
                "reported_name": second_name,
                "memory_mib": 143771,
                "compute_capability": "9.0",
                "driver_version": "570.1",
            },
        ],
        "topology_matrix": "\n".join(rows),
    }


def _nvlink_bundle():
    return CapabilityBundle(
        name="test",
        attention_backend="flex_attention",
        context_parallel_size=1,
        data_plane="transfer_engine",
        transport_protocol="nvlink_intra",
        target_output="buffered",
        target_tensor_parallel_sizes=(1,),
        inference_batch_sizes=(1,),
        max_concurrent_batches=(1,),
        requires_nvlink=True,
        gpu_direct=True,
        host_fallback_allowed=False,
    )


def test_required_nvlink_and_homogeneous_hardware_are_fail_closed():
    topology_runner._validate_hardware_for_study(_hardware(), (_nvlink_bundle(),))

    with pytest.raises(ValueError, match="all visible GPU pairs"):
        topology_runner._validate_hardware_for_study(
            _hardware(link="SYS"), (_nvlink_bundle(),)
        )
    with pytest.raises(ValueError, match="homogeneous visible GPUs"):
        topology_runner._validate_hardware_for_study(
            _hardware(second_name="NVIDIA H100"), (_nvlink_bundle(),)
        )


def test_resume_allows_operational_limits_but_not_protocol_changes(tmp_path):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.output_root = tmp_path / "session"
    runner.state_path = runner.output_root / "automatic-topology.json"
    runner.study = SimpleNamespace(
        session_id="session",
        to_dict=lambda: {"session_id": "session"},
    )
    runner.config_path = _CONFIG
    runner.hardware = _hardware()
    runner.hardware_cohort = topology_runner.hardware_cohort(runner.hardware)
    runner.formal_candidate_ids = {"candidate": "formal"}
    runner.run_policy = AutomaticRunPolicy()

    initial = runner._load_or_initialize_state()
    assert initial["resume_policy_history"] == []

    runner.run_policy = AutomaticRunPolicy(total_budget_seconds=9000.0)
    resumed = runner._load_or_initialize_state()
    assert resumed["run_policy"]["total_budget_seconds"] == 7200.0
    assert resumed["active_operational_policy"]["total_budget_seconds"] == 9000.0
    assert len(resumed["resume_policy_history"]) == 1

    runner.run_policy = AutomaticRunPolicy(confirm_repeats=2)
    with pytest.raises(ValueError, match="cannot change its measurement protocol"):
        runner._load_or_initialize_state()


def test_orphaned_formal_run_requires_exact_precompiled_plan_identity(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "attempt-01"
    run_dir.mkdir()
    (run_dir / "evidence.json").write_text("{}", encoding="utf-8")
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.formal_candidate_ids = {"candidate": "expected-formal"}
    runner.study = SimpleNamespace(
        candidate=lambda _: SimpleNamespace(candidate_id="candidate")
    )
    runner._state = {
        "attempts": [
            {
                "candidate_id": "candidate",
                "phase": "winner_confirmation",
                "state": "running",
                "run_dir": str(run_dir),
                "calibration_workload": False,
                "started_at_unix": 1.0,
                "duration_seconds": 1.0,
                "error": None,
            }
        ]
    }
    runner._save = lambda: None
    monkeypatch.setattr(
        topology_runner,
        "_active_runtime_processes",
        lambda: {"ray": [], "gpu": []},
    )
    monkeypatch.setattr(
        topology_runner,
        "load_candidate_measurement",
        lambda _: SimpleNamespace(candidate_id="partial-match-only", valid=True),
    )

    runner._recover_orphaned_attempts()

    assert runner._state["attempts"][0]["state"] == "invalid"


def test_formal_window_compares_every_short_window_equivalent(monkeypatch):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.run_policy = AutomaticRunPolicy(confirm_repeats=1)
    bundle = SimpleNamespace(
        bundle_id="bundle", name="bundle", target_tensor_parallel_sizes=(1,)
    )
    candidates = {
        candidate_id: SimpleNamespace(
            candidate_id=candidate_id,
            bundle_id=bundle.bundle_id,
            target_tensor_parallel_size=1,
        )
        for candidate_id in ("candidate-a", "candidate-b")
    }
    runner.study = SimpleNamespace(
        bundles=(bundle,),
        profile_candidate_ids=("candidate-a", "candidate-b"),
        candidate=candidates.__getitem__,
    )
    runner._state = {
        "state": "running",
        "attempts": [
            {
                "candidate_id": candidate_id,
                "phase": "role_profile",
                "state": "completed",
                "evidence_path": f"/{candidate_id}-short/evidence.json",
            }
            for candidate_id in candidates
        ],
    }
    runner._recover_orphaned_attempts = lambda: None
    runner._should_run_phase = lambda *_: False
    runner._should_run_selection_candidate = lambda *_: False
    runner._save = lambda: None
    runner._select_finalist = lambda _: {
        "recommended": {"candidate_id": "candidate-a"},
        "ranked_equivalent": [
            {"candidate_id": "candidate-a"},
            {"candidate_id": "candidate-b"},
        ],
        "environment_id": "environment",
        "hardware_cohort_id": "hardware",
        "workload_id": "short-workload",
    }
    monkeypatch.setattr(
        topology_runner,
        "load_service_observation",
        lambda _study, path: SimpleNamespace(candidate_id=path),
    )
    monkeypatch.setattr(
        topology_runner,
        "rank_topology_candidates",
        lambda *_: {
            "finalist_candidate_ids": ["candidate-a", "candidate-b"]
        },
    )

    def run_candidate(candidate, **_):
        attempt = {
            "candidate_id": candidate.candidate_id,
            "phase": "winner_confirmation",
            "state": "completed",
            "evidence_path": f"/{candidate.candidate_id}-formal/evidence.json",
        }
        runner._state["attempts"].append(attempt)
        return attempt

    runner._run_candidate = run_candidate

    def measurement(path):
        candidate_id = "candidate-a" if "candidate-a" in str(path) else "candidate-b"
        return CandidateMeasurement(
            candidate_id=f"formal-{candidate_id}",
            evidence_path=str(path),
            plan_id="plan",
            workload_id="formal-workload",
            environment_id="environment",
            hardware_cohort_id="hardware",
            committed_work_per_second=(9.0 if candidate_id == "candidate-a" else 10.0),
            optimizer_steps=3,
            peak_resident_bytes=100,
            planned_sample_pool_size=8,
            completed=True,
            finite_training_signals=True,
            lifecycle_healthy=True,
            required_capabilities_active=True,
        )

    monkeypatch.setattr(topology_runner, "load_candidate_measurement", measurement)

    state = runner._run_session()

    assert state["selection"]["short_window_recommended"]["candidate_id"] == "candidate-a"
    assert state["selection"]["recommended"]["candidate_id"] == "candidate-b"
    assert {
        row["candidate_id"]
        for row in state["selection"]["formal_window_confirmations"]
    } == {"candidate-a", "candidate-b"}


def test_attempt_propagates_normalized_hardware_label_to_evidence(
    monkeypatch, tmp_path
):
    runner = object.__new__(topology_runner.AutomaticTopologyRunner)
    runner.run_policy = AutomaticRunPolicy()
    runner.config_path = _CONFIG.resolve()
    runner.output_root = tmp_path
    runner.base_overrides = ()
    runner.hardware_cohort = {"gpu_model": "H200"}
    runner.formal_candidate_ids = {}
    runner._state = {"attempts": [], "stop_reason": None}
    runner._remaining_budget = lambda: 60.0
    runner._save = lambda: None
    runner._record_post_run_quiescence = lambda _: None
    monkeypatch.setattr(
        topology_runner,
        "_active_runtime_processes",
        lambda: {"ray": [], "gpu": []},
    )
    launched = {}

    class Process:
        pid = 123
        returncode = 1

        def wait(self, timeout):
            launched["timeout"] = timeout
            return self.returncode

    def popen(*args, **kwargs):
        launched["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(topology_runner.subprocess, "Popen", popen)
    candidate = SimpleNamespace(
        candidate_id="candidate",
        label="candidate-label",
        overrides=(),
    )

    attempt = runner._run_candidate(candidate, phase="role_profile")

    assert attempt["state"] == "failed"
    assert attempt["environment_overrides"] == {
        "ASPEC_HARDWARE_LABEL": "H200"
    }
    assert launched["env"]["ASPEC_HARDWARE_LABEL"] == "H200"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01])
def test_topology_policy_rejects_invalid_producer_headroom(value):
    with pytest.raises(ValueError, match="producer_headroom_fraction"):
        TopologyPolicy(producer_headroom_fraction=value)
