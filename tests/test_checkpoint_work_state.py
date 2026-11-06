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

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from aspec.training import checkpoint
from aspec.training.checkpoint import (
    load_checkpoint_work_state,
    prepare_checkpoint_save,
    save_checkpoint_work_state,
)
from aspec.work_lifecycle import WORK_ID_KEY, build_work_identity


def _checkpoint(
    tmp_path,
    *,
    completed_step: int = 5,
    staged: bool = True,
    world_size: int = 1,
    rng_ranks: list[int] | None = None,
):
    base = tmp_path / "checkpoints"
    step_id = completed_step + 1
    artifact = base / (
        f".iter_{step_id:07d}.incomplete" if staged else f"iter_{step_id:07d}"
    )
    for component in ("model", "optimizer", "lr_scheduler"):
        path = artifact / component
        path.mkdir(parents=True, exist_ok=True)
        (path / ".metadata").write_bytes(b"checkpoint")
    rng_dir = artifact / "rng"
    rng_dir.mkdir()
    ranks = range(world_size) if rng_ranks is None else rng_ranks
    for rank in ranks:
        torch.save(
            {
                "schema_version": checkpoint.RNG_STATE_SCHEMA_VERSION,
                "rank": rank,
                "world_size": world_size,
                "torch": torch.get_rng_state(),
                "cuda": None,
            },
            rng_dir / f"rank-{rank:05d}.pt",
        )
    (artifact / "meta.json").write_text(
        json.dumps({"global_step": completed_step, "world_size": world_size}),
        encoding="utf-8",
    )
    if not staged:
        (base / "latest_checkpointed_iteration.txt").write_text(
            str(step_id), encoding="utf-8"
        )
    return base, artifact


def _args(base):
    return Namespace(
        load_path=str(base),
        ckpt_step=None,
        continual_training=False,
        continue_source_cursor=False,
    )


def test_checkpoint_work_state_round_trip(tmp_path):
    base, staged = _checkpoint(tmp_path)
    work_ids = [
        build_work_identity(split="train", epoch=0, source_id=f"s{i}")[WORK_ID_KEY]
        for i in range(3)
    ]

    observation = save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 0,
            "dataset_size": 32,
            "dataset_fingerprint": "sha256:" + "a" * 64,
            "committed_work_ids": list(reversed(work_ids)),
        },
    )
    artifact = base / "iter_0000006"
    restored = load_checkpoint_work_state(_args(base))

    assert not staged.exists()
    assert (artifact / "complete.json").is_file()
    assert (base / "latest_checkpointed_iteration.txt").read_text() == "6"
    assert restored is not None
    assert restored["committed_work_ids"] == sorted(work_ids)
    assert restored["committed_work_count"] == 3
    assert observation["committed_work_sha256"] == restored["committed_work_sha256"]
    metadata = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
    assert metadata["work_state"]["committed_work_count"] == 3


def test_exact_resume_can_select_an_explicit_published_checkpoint(tmp_path):
    base, _ = _checkpoint(tmp_path)
    save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 1,
            "dataset_size": 32,
            "dataset_fingerprint": "sha256:" + "a" * 64,
            "committed_work_ids": [],
        },
    )
    selected = base / "iter_0000006"
    # Selecting a published frontier must not depend on the parent's tracker.
    (base / "latest_checkpointed_iteration.txt").write_text("999")
    args = _args(selected)
    assert checkpoint.resolve_resume_model_dir(args) == selected / "model"
    assert load_checkpoint_work_state(args)["optimizer_step"] == 5
    args.ckpt_step = 7
    with pytest.raises(ValueError, match="disagrees"):
        checkpoint.resolve_resume_model_dir(args)


def test_explicit_checkpoint_cannot_resolve_a_missing_model(tmp_path):
    directory = tmp_path / "iter_0000006"
    directory.mkdir()
    assert checkpoint.resolve_resume_model_dir(_args(directory)) is None


@pytest.mark.parametrize(
    "field,value", [("dataset_seed", 99), ("shuffle_dataset", False), ("length_group_size", 64)]
)
def test_exact_resume_preserves_source_order_independently_of_training_seed(tmp_path, field, value):
    base, _ = _checkpoint(tmp_path)
    order = {"seed": 7, "shuffle": True, "length_group_size": 32, "packing": False}
    save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 2,
            "dataset_size": 32,
            "dataset_fingerprint": "sha256:" + "a" * 64,
            "committed_work_ids": ["work-1"],
            "dataset_order": order,
        },
    )
    args = _args(base)
    args.seed, args.dataset_seed = 123, 7
    assert load_checkpoint_work_state(args)["dataset_order"] == order
    setattr(args, field, value)
    with pytest.raises(RuntimeError, match="cannot change dataset ordering"):
        load_checkpoint_work_state(args)


def test_checkpoint_work_state_detects_tampering(tmp_path):
    base, _ = _checkpoint(tmp_path)
    save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 0,
            "dataset_size": 1,
            "dataset_fingerprint": "sha256:" + "b" * 64,
            "committed_work_ids": ["work-1"],
        },
    )
    artifact = base / "iter_0000006"
    path = artifact / "work_state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["committed_work_ids"].append("work-2")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="count mismatch"):
        load_checkpoint_work_state(_args(base))


def test_checkpoint_completion_marker_detects_tampering(tmp_path):
    base, _ = _checkpoint(tmp_path)
    save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 0,
            "dataset_size": 1,
            "dataset_fingerprint": "sha256:" + "d" * 64,
            "committed_work_ids": ["work-1"],
        },
    )
    marker_path = base / "iter_0000006" / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["committed_work_count"] = 2
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="completion marker disagrees"):
        load_checkpoint_work_state(_args(base))


def test_checkpoint_publication_requires_rng_state_for_every_rank(tmp_path):
    base, staged = _checkpoint(tmp_path, world_size=2, rng_ranks=[0])

    with pytest.raises(RuntimeError, match=r"RNG/work state; missing=.*rank-00001\.pt"):
        save_checkpoint_work_state(
            str(base),
            5,
            {
                "dataset_epoch": 0,
                "dataset_size": 1,
                "dataset_fingerprint": "sha256:" + "e" * 64,
                "committed_work_ids": ["work-1"],
            },
        )

    assert staged.is_dir()
    assert not (staged / "complete.json").exists()
    assert not (base / "iter_0000006").exists()
    assert not (base / "latest_checkpointed_iteration.txt").exists()


def test_rank_rng_state_is_selected_by_rank(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000001"
    rng_dir = checkpoint_dir / "rng"
    rng_dir.mkdir(parents=True)
    for rank, value in ((0, 3), (1, 7)):
        torch.save(
            {
                "schema_version": checkpoint.RNG_STATE_SCHEMA_VERSION,
                "rank": rank,
                "world_size": 2,
                "torch": torch.tensor([value], dtype=torch.uint8),
                "cuda": None,
            },
            rng_dir / f"rank-{rank:05d}.pt",
        )

    state = checkpoint._load_rank_rng_state(checkpoint_dir, rank=1, world_size=2)

    assert state["rank"] == 1
    assert state["torch"].tolist() == [7]


def test_legacy_rank_zero_rng_is_rejected_for_multi_rank_resume(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000001"
    checkpoint_dir.mkdir()
    torch.save({"torch": torch.get_rng_state()}, checkpoint_dir / "rng.pt")

    with pytest.raises(RuntimeError, match="one RNG checkpoint per rank"):
        checkpoint._load_rank_rng_state(checkpoint_dir, rank=1, world_size=2)


def test_legacy_rank_zero_rng_remains_readable_for_single_rank_resume(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000001"
    checkpoint_dir.mkdir()
    torch.save(
        {"torch": torch.tensor([9], dtype=torch.uint8), "cuda": None},
        checkpoint_dir / "rng.pt",
    )

    state = checkpoint._load_rank_rng_state(checkpoint_dir, rank=0, world_size=1)

    assert state["torch"].tolist() == [9]


def test_true_resume_rejects_legacy_checkpoint_without_work_state(tmp_path):
    base, _ = _checkpoint(tmp_path, staged=False)

    with pytest.raises(RuntimeError, match="exact resume"):
        load_checkpoint_work_state(_args(base))


def test_continual_training_does_not_claim_exact_resume(tmp_path):
    base, _ = _checkpoint(tmp_path, staged=False)
    args = _args(base)
    args.continual_training = True

    assert load_checkpoint_work_state(args) is None


def test_continual_training_can_restore_only_the_source_cursor(tmp_path):
    base, _ = _checkpoint(tmp_path)
    work_ids = [
        build_work_identity(split="train", epoch=0, source_id="s0")[WORK_ID_KEY]
    ]
    save_checkpoint_work_state(
        str(base),
        5,
        {
            "dataset_epoch": 0,
            "dataset_size": 32,
            "dataset_fingerprint": "sha256:" + "a" * 64,
            "committed_work_ids": work_ids,
        },
    )
    args = _args(base)
    args.continual_training = True
    args.continue_source_cursor = True

    restored = load_checkpoint_work_state(args)

    assert restored is not None
    assert restored["optimizer_step"] == 5
    assert restored["committed_work_ids"] == work_ids


def test_continual_training_rebuilds_current_master_params_without_old_optimizer(
    monkeypatch,
):
    optimizer = mock.Mock()
    optimizer.sync_fp32_params_from_model = mock.Mock()
    actor = SimpleNamespace(
        args=Namespace(continual_training=True),
        optimizer=optimizer,
    )
    old_optimizer_load = mock.Mock(
        side_effect=AssertionError("continual training must not read old optimizer state")
    )
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", old_optimizer_load)
    monkeypatch.setattr("aspec.training.checkpoint.torch.cuda.synchronize", mock.Mock())
    monkeypatch.setattr("aspec.training.checkpoint.dist.barrier", mock.Mock())

    checkpoint.finalize_load(
        actor,
        {
            "rng": None,
            "metadata": {},
            "iteration": 3,
            "optimizer_dir": "/an/old/stage/optimizer",
        },
    )

    optimizer.sync_fp32_params_from_model.assert_called_once_with()
    old_optimizer_load.assert_not_called()


def test_failed_publication_keeps_previous_latest_checkpoint(tmp_path):
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "latest_checkpointed_iteration.txt").write_text("4", encoding="utf-8")
    _, staged = _checkpoint(tmp_path, completed_step=5)

    with pytest.raises(ValueError, match="dataset fingerprint"):
        save_checkpoint_work_state(
            str(base),
            5,
            {
                "dataset_epoch": 0,
                "dataset_size": 1,
                "dataset_fingerprint": "not-a-digest",
                "committed_work_ids": ["work-1"],
            },
        )

    assert staged.is_dir()
    assert not (base / "iter_0000006").exists()
    assert (base / "latest_checkpointed_iteration.txt").read_text() == "4"


def test_prepare_checkpoint_save_rejects_stale_staging_directory(tmp_path):
    base = tmp_path / "checkpoints"
    staged = base / ".iter_0000006.incomplete"
    staged.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="staging checkpoint"):
        prepare_checkpoint_save(str(base), completed_step=5)


def test_publication_rejects_metadata_from_a_different_optimizer_step(tmp_path):
    base, staged = _checkpoint(tmp_path, completed_step=5)
    (staged / "meta.json").write_text(
        json.dumps({"global_step": 4}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="metadata does not match"):
        save_checkpoint_work_state(
            str(base),
            5,
            {
                "dataset_epoch": 0,
                "dataset_size": 1,
                "dataset_fingerprint": "sha256:" + "c" * 64,
                "committed_work_ids": ["work-1"],
            },
        )

    assert staged.is_dir()
    assert not (base / "latest_checkpointed_iteration.txt").exists()


def _load_actor(base, *, continual_training: bool = False):
    return SimpleNamespace(
        args=Namespace(
            load_path=str(base),
            ckpt_step=None,
            continual_training=continual_training,
            world_size=1,
        ),
        model=mock.MagicMock(),
        optimizer=mock.MagicMock(),
        lr_scheduler=mock.MagicMock(),
    )


def test_checkpoint_model_load_failure_is_fatal(tmp_path, monkeypatch):
    base, _ = _checkpoint(tmp_path, staged=False)
    actor = _load_actor(base, continual_training=True)
    monkeypatch.setattr(
        "aspec.training.checkpoint.dcp.load",
        mock.Mock(side_effect=RuntimeError("corrupt shard")),
    )

    with pytest.raises(RuntimeError, match="Failed to load model"):
        checkpoint.load(actor)


def test_exact_resume_rejects_world_size_change_before_loading_shards(
    tmp_path, monkeypatch
):
    base, _ = _checkpoint(tmp_path, staged=False)
    actor = _load_actor(base)
    actor.args.world_size = 2
    load_mock = mock.Mock()
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", load_mock)

    with pytest.raises(RuntimeError, match="world-size changes"):
        checkpoint.load(actor)

    load_mock.assert_not_called()


def test_exact_resume_rejects_logical_parallelism_change(tmp_path, monkeypatch):
    base, artifact = _checkpoint(tmp_path, staged=False)
    metadata_path = artifact / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_topology"] = {
        "world_size": 1,
        "draft_context_parallel_size": 2,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    actor = _load_actor(base)
    actor.args.draft_context_parallel_size = 1
    load_mock = mock.Mock()
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", load_mock)

    with pytest.raises(RuntimeError, match="logical training-topology changes"):
        checkpoint.load(actor)

    load_mock.assert_not_called()


def test_exact_resume_requires_initialized_optimizer(tmp_path, monkeypatch):
    base, _ = _checkpoint(tmp_path, staged=False)
    actor = _load_actor(base)
    actor.optimizer = None
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", mock.Mock())

    with pytest.raises(RuntimeError, match="initialized optimizer"):
        checkpoint.load(actor)


def test_exact_resume_requires_initialized_lr_scheduler(tmp_path, monkeypatch):
    base, _ = _checkpoint(tmp_path, staged=False)
    actor = _load_actor(base)
    actor.lr_scheduler = None
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", mock.Mock())

    with pytest.raises(RuntimeError, match="initialized LR scheduler"):
        checkpoint.load(actor)


def test_exact_resume_requires_lr_scheduler_state(tmp_path, monkeypatch):
    base, artifact = _checkpoint(tmp_path, staged=False)
    (artifact / "lr_scheduler" / ".metadata").unlink()
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", mock.Mock())

    with pytest.raises(RuntimeError, match="LR scheduler checkpoint"):
        checkpoint.load(_load_actor(base))


def test_exact_resume_requires_rng_state(tmp_path, monkeypatch):
    base, artifact = _checkpoint(tmp_path, staged=False)
    (artifact / "rng" / "rank-00000.pt").unlink()
    monkeypatch.setattr("aspec.training.checkpoint.dcp.load", mock.Mock())

    with pytest.raises(RuntimeError, match="RNG checkpoint"):
        checkpoint.load(_load_actor(base))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("bad_state", ["missing", "none", "dtype", "rank", "empty"])
def test_exact_resume_rejects_invalid_cpu_rng_payload(tmp_path, legacy, bad_state):
    directory = tmp_path / "iter_0000001"
    (directory / "rng").mkdir(parents=True)
    state = {
        "schema_version": checkpoint.RNG_STATE_SCHEMA_VERSION,
        "rank": 0,
        "world_size": 1,
        "cuda": None,
    }
    if bad_state != "missing":
        state["torch"] = {
            "none": None,
            "dtype": torch.zeros(16, dtype=torch.float32),
            "rank": torch.zeros(2, 8, dtype=torch.uint8),
            "empty": torch.empty(0, dtype=torch.uint8),
        }[bad_state]
    path = directory / ("rng.pt" if legacy else "rng/rank-00000.pt")
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="CPU RNG state"):
        checkpoint._load_rank_rng_state(directory, rank=0, world_size=1)


@pytest.mark.parametrize("legacy", [False, True])
def test_cuda_resume_requires_saved_cuda_rng(tmp_path, monkeypatch, legacy):
    directory = tmp_path / "iter_0000001"
    (directory / "rng").mkdir(parents=True)
    state = {
        "schema_version": checkpoint.RNG_STATE_SCHEMA_VERSION,
        "rank": 0,
        "world_size": 1,
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    torch.save(state, directory / ("rng.pt" if legacy else "rng/rank-00000.pt"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA RNG state"):
        checkpoint._load_rank_rng_state(directory, rank=0, world_size=1)


def test_exact_resume_finalize_cannot_silently_skip_missing_rng(monkeypatch):
    actor = SimpleNamespace(args=Namespace(continual_training=False))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    with pytest.raises(RuntimeError, match="RNG state"):
        checkpoint.finalize_load(actor, {"rng": None})
