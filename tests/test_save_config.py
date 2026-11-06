# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_save_config.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for saving resolved config to output_dir."""

import os
from pathlib import Path

import pytest
from omegaconf import MISSING, OmegaConf

from aspec.config.train_config import Config, load_config, save_config

_DRAFT_CONFIG = Path(__file__).parents[1] / "aspec" / "config" / "dspark_draft_config.json"

# ---------------------------------------------------------------------------
# Unit tests for save_config / load_config round-trip
# ---------------------------------------------------------------------------


def test_save_resolved_config_basic(tmp_path):
    """Config with output_dir saves to output_dir/config.yaml and round-trips correctly."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = OmegaConf.structured(Config)
    config.output_dir = str(output_dir)
    config.model.target_model_path = "test-model"
    config.dataset.train_data_path = "test-data"
    config.training.micro_batch_size = 8

    save_path = str(output_dir / "config.yaml")
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.output_dir == str(output_dir)
    assert reloaded.model.target_model_path == "test-model"
    assert reloaded.dataset.train_data_path.endswith("test-data")
    assert reloaded.training.micro_batch_size == 8


def test_save_resolved_config_with_cli_overrides(tmp_path):
    """CLI dotlist overrides are persisted in the saved config."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base = OmegaConf.structured(Config)
    base.output_dir = str(output_dir)
    base.model.target_model_path = "base-model"
    base.dataset.train_data_path = "base-data"

    config = load_config(
        cli_args=["training.micro_batch_size=16", "training.learning_rate=3e-5"],
        base_config=base,
    )

    save_path = str(output_dir / "config.yaml")
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.training.micro_batch_size == 16
    assert reloaded.training.learning_rate == 3e-5
    assert reloaded.model.target_model_path == "base-model"


def test_save_config_materializes_environment_interpolations(tmp_path, monkeypatch):
    monkeypatch.setenv("ASPEC_TEST_CACHE", str(tmp_path / "cache"))
    config = OmegaConf.structured(Config)
    config.cache_dir = "${oc.env:ASPEC_TEST_CACHE}"
    save_path = tmp_path / "config.yaml"

    save_config(config, str(save_path))
    monkeypatch.delenv("ASPEC_TEST_CACHE")

    reloaded = OmegaConf.load(save_path)
    assert reloaded.cache_dir == str(tmp_path / "cache")


def test_save_resolved_config_empty_output_dir():
    """When output_dir is empty, the OmegaConf.select guard evaluates to falsy."""
    config = OmegaConf.structured(Config)
    config.output_dir = ""

    assert not OmegaConf.select(config, "output_dir")


def test_save_resolved_config_missing_output_dir():
    """When output_dir is MISSING (???), OmegaConf.select returns None."""
    config = OmegaConf.structured(Config)
    config.output_dir = MISSING

    assert OmegaConf.select(config, "output_dir") is None


def test_load_config_hub_dataset_id_not_absolutized(tmp_path):
    """Hub dataset ids should remain unchanged after config load."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        "\n".join(
            [
                "model:",
                "  target_model_path: test-model",
                "dataset:",
                "  train_data_path: HuggingFaceH4/ultrachat_200k",
                "training:",
                "  training_num_gpus_per_node: 1",
                "  training_num_nodes: 1",
            ]
        )
    )

    config = load_config(config_path=str(cfg))
    assert config.dataset.train_data_path == "HuggingFaceH4/ultrachat_200k"


def test_load_config_no_snapshot_by_default(tmp_path):
    """load_config() should not write output_dir/config.yaml unless requested."""
    out_dir = tmp_path / "out"
    cfg = tmp_path / "nosnapshot.yaml"
    cfg.write_text(
        "\n".join(
            [
                "model:",
                "  target_model_path: test-model",
                "dataset:",
                "  train_data_path: test-data",
                "training:",
                "  training_num_gpus_per_node: 1",
                "  training_num_nodes: 1",
                f"output_dir: '{out_dir}'",
            ]
        )
    )

    load_config(config_path=str(cfg))
    assert not (out_dir / "config.yaml").exists()

    load_config(config_path=str(cfg), save_snapshot=True)
    assert (out_dir / "config.yaml").exists()


def test_save_config_overwrites_existing(tmp_path):
    """Saving config twice overwrites the previous file."""
    config = OmegaConf.structured(Config)
    config.output_dir = str(tmp_path)
    config.training.micro_batch_size = 4

    save_path = str(tmp_path / "config.yaml")
    save_config(config, save_path)

    config.training.micro_batch_size = 8
    save_config(config, save_path)

    reloaded = load_config(config_path=save_path)
    assert reloaded.training.micro_batch_size == 8


# ---------------------------------------------------------------------------
# Integration tests for parse_config()
# ---------------------------------------------------------------------------


def _write_minimal_config(path, output_dir="", **overrides):
    """Write a minimal valid YAML config to *path*."""
    lines = [
        "model:",
        "  target_model_path: Qwen/Qwen3-8B",
        f"  draft_model_config: '{_DRAFT_CONFIG}'",
        "dataset:",
        "  train_data_path: test-data",
        "  min_loss_tokens: 32",
        "training:",
        "  training_num_gpus_per_node: 1",
        "  training_num_nodes: 1",
        "inference:",
        "  inference_num_gpus: 1",
        "  sglang:",
        "    tp_size: 1",
        f"output_dir: '{output_dir}'",
    ]
    for dotkey, val in overrides.items():
        # Only supports top-level overrides for simplicity
        lines.append(f"{dotkey}: {val}")
    path.write_text("\n".join(lines))


def test_parse_config_saves_to_output_dir(tmp_path, monkeypatch):
    """parse_config() saves config.yaml into output_dir and returns valid flat_args."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    args = parse_config()

    assert (output_dir / "config.yaml").exists()
    assert args.target_model_path == "Qwen/Qwen3-8B"
    assert (output_dir / "plan.json").exists()

    reloaded = load_config(config_path=str(output_dir / "config.yaml"))
    assert reloaded.output_dir == str(output_dir)


def test_parse_config_creates_nested_dir(tmp_path, monkeypatch):
    """parse_config() creates nested output_dir automatically."""
    output_dir = tmp_path / "a" / "b" / "c"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    parse_config()

    assert (output_dir / "config.yaml").exists()


def test_parse_config_rejects_empty_output_dir(tmp_path, monkeypatch):
    """A real run must have a durable artifact directory."""
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir="")

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    with pytest.raises(ValueError, match="output_dir is required"):
        parse_config()

    assert not list(tmp_path.glob("**/config.yaml"))


def test_parse_config_allows_unrelated_files_in_output_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "operator-note.txt").write_text("keep me", encoding="utf-8")
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))
    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    parse_config()

    assert (output_dir / "operator-note.txt").read_text(encoding="utf-8") == "keep me"
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "plan.json").is_file()


@pytest.mark.parametrize("artifact", ["evidence.json", "tracking.json"])
def test_parse_config_rejects_managed_artifact_conflict(tmp_path, monkeypatch, artifact):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / artifact).write_text("{}\n", encoding="utf-8")
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))
    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    with pytest.raises(FileExistsError, match="ASpec-managed"):
        parse_config()
    assert (output_dir / artifact).read_text(encoding="utf-8") == "{}\n"
    assert not (output_dir / "config.yaml").exists()


def test_parse_config_print_config_only_exits(tmp_path, monkeypatch):
    """--print-config-only prints config and exits."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file), "--print-config-only"])

    from aspec.train_entry import parse_config

    with pytest.raises(SystemExit) as exc_info:
        parse_config()

    assert exc_info.value.code == 0
    assert not output_dir.exists()


def test_parse_config_cli_overrides_saved(tmp_path, monkeypatch):
    """CLI overrides are reflected in the saved config.yaml."""
    output_dir = tmp_path / "out"
    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--config", str(config_file), "training.micro_batch_size=32"],
    )

    from aspec.train_entry import parse_config

    args = parse_config()

    assert args.micro_batch_size == 32

    reloaded = load_config(config_path=str(output_dir / "config.yaml"))
    assert reloaded.training.micro_batch_size == 32


@pytest.mark.skipif(
    getattr(os, "getuid", lambda: -1)() == 0, reason="root ignores file permissions"
)
def test_parse_config_rejects_readonly_output_dir(tmp_path, monkeypatch):
    """A run cannot start if its required artifacts cannot be persisted."""
    output_dir = tmp_path / "readonly"
    output_dir.mkdir()
    os.chmod(str(output_dir), 0o444)

    config_file = tmp_path / "test.yaml"
    _write_minimal_config(config_file, output_dir=str(output_dir))

    monkeypatch.setattr("sys.argv", ["prog", "--config", str(config_file)])

    from aspec.train_entry import parse_config

    try:
        with pytest.raises(OSError):
            parse_config()
    finally:
        os.chmod(str(output_dir), 0o755)
    # Verify no config.yaml was written (check after restoring permissions)
    assert not (output_dir / "config.yaml").exists()
