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

"""Prepare and audit native SpecForge end-to-end benchmark runs.

The runner does not patch SpecForge or replace its training loop.  It converts
ASpec's compact loss-mask representation to SpecForge's documented
pre-tokenized JSONL boundary, launches the upstream CLI, and derives elapsed
training time from independent optimizer-completion observations. Native
per-optimizer-step performance metrics remain separately recorded diagnostics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STEP = re.compile(r"step\s+(\d+):\s+(\{.*?\})(?=\s*step\s+\d+:|\s*$)")
_INITIAL_KEYMAP = {
    "context_proj.weight": "fc.weight",
    "context_norm.weight": "hidden_norm.weight",
    "final_norm.weight": "norm.weight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_loss_mask(value: str, *, expected_length: int) -> list[int]:
    if not isinstance(value, str) or not value:
        raise ValueError("packed_loss_mask must be a non-empty comma-separated string")
    try:
        segments = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("packed_loss_mask contains a non-integer segment") from error
    if any(segment < 0 for segment in segments):
        raise ValueError("packed_loss_mask segments must be non-negative")
    if sum(segments) != expected_length:
        raise ValueError(
            "packed_loss_mask length does not match input_ids: "
            f"{sum(segments)} != {expected_length}"
        )
    mask: list[int] = []
    for index, segment in enumerate(segments):
        mask.extend([index % 2] * segment)
    return mask


def prepare_data(source: Path, output: Path, manifest_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    rows = 0
    input_tokens = 0
    supervised_positions = 0
    lengths: list[int] = []
    identities: set[str] = set()
    identity_digest = hashlib.sha256()
    try:
        with (
            source.open(encoding="utf-8") as reader,
            temporary.open("x", encoding="utf-8") as writer,
        ):
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on source line {line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"source line {line_number} is not an object")
                identity = record.get("data_id")
                if not isinstance(identity, str) or not identity:
                    raise ValueError(f"source line {line_number} has no data_id")
                if identity in identities:
                    raise ValueError(f"duplicate data_id {identity!r}")
                identities.add(identity)

                tokens = record.get("input_ids")
                if (
                    not isinstance(tokens, list)
                    or not tokens
                    or any(
                        not isinstance(token, int) or isinstance(token, bool) for token in tokens
                    )
                ):
                    raise ValueError(f"source line {line_number} has invalid input_ids")
                mask = unpack_loss_mask(record.get("packed_loss_mask"), expected_length=len(tokens))
                observed_loss = sum(mask)
                declared_loss = record.get("loss_tokens")
                if declared_loss is not None and int(declared_loss) != observed_loss:
                    raise ValueError(
                        f"source line {line_number} loss_tokens disagrees with the mask"
                    )
                if observed_loss < 1:
                    raise ValueError(f"source line {line_number} has no supervised positions")

                writer.write(
                    json.dumps(
                        {"input_ids": tokens, "loss_mask": mask},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                identity_digest.update(identity.encode("utf-8"))
                identity_digest.update(b"\0")
                rows += 1
                input_tokens += len(tokens)
                supervised_positions += observed_loss
                lengths.append(len(tokens))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if not rows:
        output.unlink(missing_ok=True)
        raise ValueError("source contains no records")
    manifest = {
        "schema": "aspec.specforge-pretokenized/v1",
        "conversion": "packed_loss_mask -> exact binary loss_mask; order unchanged",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "rows": rows,
        "input_tokens": input_tokens,
        "supervised_positions": supervised_positions,
        "length": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": mean(lengths),
        },
        "ordered_data_id_sha256": identity_digest.hexdigest(),
    }
    manifest_path = manifest_path or output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_initial(source: Path, draft_config: Path, output: Path) -> dict:
    """Adapt only the three renamed DSpark backbone tensors for SpecForge.

    The model math and every tensor value stay unchanged.  SpecForge's native
    warm-start loader reports any remaining structural mismatch during the
    subsequent smoke run.
    """

    if output.exists():
        raise FileExistsError(output)
    source_weights = source / "model.safetensors"
    if not source_weights.is_file() or not draft_config.is_file():
        raise FileNotFoundError("source weights and SpecForge draft config are required")
    from safetensors.torch import load_file, save_file

    state = load_file(str(source_weights), device="cpu")
    missing = sorted(set(_INITIAL_KEYMAP) - set(state))
    collisions = sorted(set(_INITIAL_KEYMAP.values()).intersection(state))
    if missing or collisions:
        raise ValueError(
            f"initial checkpoint key mismatch: missing={missing}, collisions={collisions}"
        )
    adapted = {
        _INITIAL_KEYMAP.get(name, name): tensor.contiguous() for name, tensor in state.items()
    }
    output.mkdir(parents=True)
    try:
        save_file(adapted, str(output / "model.safetensors"))
        shutil.copy2(draft_config, output / "config.json")
        manifest = {
            "schema": "aspec.specforge-initial-adapter/v1",
            "source": str(source.resolve()),
            "source_weights_sha256": sha256(source_weights),
            "draft_config": str(draft_config.resolve()),
            "draft_config_sha256": sha256(draft_config),
            "keymap": _INITIAL_KEYMAP,
            "tensor_count": len(adapted),
            "output_weights_sha256": sha256(output / "model.safetensors"),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _command_identity(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _source_identity(checkout: Path) -> dict:
    return {
        "root": str(checkout.resolve()),
        "revision": _command_identity(["git", "-C", str(checkout), "rev-parse", "HEAD"]),
        "status": _command_identity(["git", "-C", str(checkout), "status", "--short"]),
    }


def _system_identity() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _command_identity([sys.executable, "-m", "pip", "freeze", "--all"]),
        "nvidia_smi": _command_identity(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _native_train_command(config: Path, overrides: list[str]) -> list[str]:
    # The subprocess runs from the upstream checkout, not ASpec's cwd.
    return [
        sys.executable,
        "-m",
        "specforge.cli",
        "train",
        "--config",
        str(config.resolve()),
        *overrides,
    ]


def _parse_metrics(lines: list[str]) -> dict[int, list[dict]]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for line in lines:
        clean_line = _ANSI_ESCAPE.sub("", line.strip())
        for match in _STEP.finditer(clean_line):
            payload = ast.literal_eval(match.group(2))
            if not isinstance(payload, dict):
                raise ValueError("SpecForge step metric is not a dictionary")
            by_step[int(match.group(1))].append(payload)
    return dict(by_step)


def summarize_metrics(
    metrics: dict[int, list[dict]],
    *,
    expected_steps: int,
    expected_ranks: int,
    expected_documents: int,
    expected_input_tokens: int,
    expected_supervised_positions: int,
    warmup_steps: int,
) -> dict:
    expected = list(range(1, expected_steps + 1))
    if sorted(metrics) != expected:
        raise ValueError(
            f"optimizer steps are incomplete: observed={sorted(metrics)}, expected={expected}"
        )
    if not 0 <= warmup_steps < expected_steps:
        raise ValueError("warmup_steps must leave at least one measured step")

    critical_rows = []
    sample_total = 0
    for step in expected:
        rows = metrics[step]
        if len(rows) != expected_ranks:
            raise ValueError(f"step {step} has {len(rows)} rank metrics, expected {expected_ranks}")
        try:
            row = max(rows, key=lambda item: float(item["perf/optimizer_step_time_s"]))
            elapsed = float(row["perf/optimizer_step_time_s"])
            rate = float(row["perf/global_samples_per_second"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"step {step} lacks native performance metrics") from error
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"step {step} has invalid native performance metrics")
        samples = round(elapsed * rate)
        if not math.isclose(elapsed * rate, samples, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"step {step} reports a non-integral global sample count")
        sample_total += samples
        critical_rows.append({"step": step, "seconds": elapsed, "samples": samples})
    if sample_total != expected_documents:
        raise ValueError(
            f"native metrics cover {sample_total} documents, expected {expected_documents}"
        )

    full_seconds = sum(item["seconds"] for item in critical_rows)
    measured = critical_rows[warmup_steps:]
    measured_seconds = sum(item["seconds"] for item in measured)
    measured_documents = sum(item["samples"] for item in measured)
    document_fraction = measured_documents / expected_documents
    estimated_input_tokens = expected_input_tokens * document_fraction
    estimated_supervised = expected_supervised_positions * document_fraction
    return {
        "time_boundary": (
            "sum of upstream SpecForge perf/optimizer_step_time_s, selecting the "
            "slowest DP rank at every synchronized optimizer boundary"
        ),
        "full": {
            "steps": expected_steps,
            "documents": sample_total,
            "seconds": full_seconds,
            "documents_per_second": expected_documents / full_seconds,
            "input_tokens_per_second": expected_input_tokens / full_seconds,
            "supervised_positions_per_second": expected_supervised_positions / full_seconds,
        },
        "steady": {
            "warmup_steps": warmup_steps,
            "steps": len(measured),
            "documents": measured_documents,
            "seconds": measured_seconds,
            "documents_per_second": measured_documents / measured_seconds,
            "token_rate_estimation": (
                "estimated from the full-corpus mean tokens per document because "
                "upstream step metrics do not expose per-sample token counts"
            ),
            "estimated_input_tokens_per_second": (estimated_input_tokens / measured_seconds),
            "estimated_supervised_positions_per_second": (estimated_supervised / measured_seconds),
        },
        "critical_rank_steps": critical_rows,
    }


def summarize_wall_observations(
    directory: Path, *, expected_steps: int, expected_ranks: int,
    expected_documents: int, expected_input_tokens: int, warmup_steps: int,
) -> dict:
    """Exact committed tokens over one continuous, single-host monotonic clock."""
    if not 0 <= warmup_steps < expected_steps:
        raise ValueError("warmup_steps must leave at least one measured step")
    paths = sorted(directory.glob("rank-*.jsonl"))
    if len(paths) != expected_ranks:
        raise ValueError("Incomplete wall-clock rank observations")
    starts, hosts, ranks, by_step, seen = [], set(), set(), defaultdict(list), set()
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        start, *steps = rows
        rank = start["rank"]
        if start["event"] != "loop_start" or start["clock"] != "perf_counter_ns" or rank in ranks:
            raise ValueError("Invalid or duplicate wall-clock rank start")
        ranks.add(rank)
        hosts.add(start["hostname"])
        starts.append(start["clock_ns"])
        if [row["step"] for row in steps] != list(range(1, expected_steps + 1)):
            raise ValueError("Incomplete optimizer completion sequence")
        previous = start["clock_ns"]
        for row in steps:
            stamp = row["clock_ns"]
            if (row["event"] != "optimizer_complete" or row["rank"] != rank
                    or not isinstance(stamp, int) or stamp <= previous):
                raise ValueError("Invalid optimizer completion clock")
            previous = stamp
            ids = row["sample_ids"]
            if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Duplicate or empty committed native sample IDs")
            seen.update(ids)
            if not isinstance(row["input_tokens"], int) or row["input_tokens"] <= 0:
                raise ValueError("Invalid observed input-token count")
            by_step[row["step"]].append(row)
    if len(hosts) != 1 or ranks != set(range(expected_ranks)):
        raise ValueError("Wall observations require complete ranks on one monotonic-clock host")
    records = [{"step": step, "completed_at_ns": max(r["clock_ns"] for r in rows),
                "documents": sum(len(r["sample_ids"]) for r in rows),
                "input_tokens": sum(r["input_tokens"] for r in rows)}
               for step, rows in sorted(by_step.items())]
    if any(b["completed_at_ns"] <= a["completed_at_ns"] for a, b in zip(records, records[1:])):
        raise ValueError("Non-monotonic global optimizer completion clock")
    documents = sum(r["documents"] for r in records)
    tokens = sum(r["input_tokens"] for r in records)
    if (documents, tokens) != (expected_documents, expected_input_tokens):
        raise ValueError("Observed complete-corpus coverage does not match expected work")

    def window(rows, start):
        seconds = (rows[-1]["completed_at_ns"] - start) / 1e9
        docs = sum(r["documents"] for r in rows)
        count = sum(r["input_tokens"] for r in rows)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid continuous wall-clock duration")
        return {"steps": len(rows), "documents": docs, "input_tokens": count,
                "seconds": seconds, "documents_per_second": docs / seconds,
                "input_tokens_per_second": count / seconds}

    start = min(starts)
    return {
        "time_boundary": "continuous training loop to last optimizer completion; no per-step summation",
        "clock": "single-host perf_counter_ns", "hostname": next(iter(hosts)),
        "full": window(records, start),
        "steady": {"warmup_steps": warmup_steps, **window(
            records[warmup_steps:], records[warmup_steps - 1]["completed_at_ns"] if warmup_steps else start)},
        "optimizer_completions": records,
    }


def _prepare_wall_observer(args) -> tuple[Path, dict]:
    directory = args.result.with_suffix(".wall")
    directory.mkdir()
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    shutil.copyfile(Path(__file__).with_name("specforge_wall_observer.py"),
                    bootstrap / "_aspec_specforge_wall_observer.py")
    (bootstrap / "sitecustomize.py").write_text(
        "from _aspec_specforge_wall_observer import install\ninstall()\n", encoding="utf-8")
    environment = dict(os.environ)
    # Only observation shims live here: no aspec/specforge/sglang/torch package.
    # The installed frameworks still resolve through their isolated venv.
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(bootstrap.resolve()), environment.get("PYTHONPATH")]))
    environment["ASPEC_SPECFORGE_OBSERVATION_DIR"] = str(directory.resolve())
    return directory, environment


def run(args) -> dict:
    if args.result.exists() or args.log.exists():
        raise FileExistsError("result and log paths must be fresh")
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = _native_train_command(args.config, args.overrides)
    observation_dir, environment = (None, dict(os.environ))
    if not args.native_timers_only:
        observation_dir, environment = _prepare_wall_observer(args)
    lines: list[str] = []
    with args.log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            lines.append(line.rstrip("\n"))
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

    summary = summarize_metrics(
        _parse_metrics(lines),
        expected_steps=args.expected_steps,
        expected_ranks=args.expected_ranks,
        expected_documents=args.expected_documents,
        expected_input_tokens=args.expected_input_tokens,
        expected_supervised_positions=args.expected_supervised_positions,
        warmup_steps=args.warmup_steps,
    )
    native_metrics = summary
    if observation_dir is not None:
        summary = summarize_wall_observations(
            observation_dir, expected_steps=args.expected_steps, expected_ranks=args.expected_ranks,
            expected_documents=args.expected_documents, expected_input_tokens=args.expected_input_tokens,
            warmup_steps=args.warmup_steps,
        )
    result = {
        "schema": "aspec.specforge-e2e-benchmark/v2",
        "framework": "specforge",
        "native_cli": command,
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "source_control": _source_identity(args.checkout),
        "system": _system_identity(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "MC_USE_NVLINK_IPC",
                "MC_FORCE_MNNVL",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "corpus": {
            "documents": args.expected_documents,
            "input_tokens": args.expected_input_tokens,
            "supervised_positions": args.expected_supervised_positions,
        },
        "metrics": summary,
        "native_metrics": native_metrics,
        "wall_observation": ({"path": str(observation_dir.resolve()),
                              "bootstrap_only_pythonpath": environment["PYTHONPATH"]}
                             if observation_dir is not None else None),
        "log": {"path": str(args.log.resolve()), "sha256": sha256(args.log)},
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path)

    initial = commands.add_parser("prepare-initial")
    initial.add_argument("--source", type=Path, required=True)
    initial.add_argument("--draft-config", type=Path, required=True)
    initial.add_argument("--output", type=Path, required=True)

    execute = commands.add_parser("run")
    execute.add_argument("--checkout", type=Path, required=True)
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--result", type=Path, required=True)
    execute.add_argument("--log", type=Path, required=True)
    execute.add_argument("--expected-steps", type=int, required=True)
    execute.add_argument("--native-timers-only", action="store_true",
                         help="Legacy native-timer reference; cannot supply unified wall-clock ratios")
    execute.add_argument("--expected-ranks", type=int, required=True)
    execute.add_argument("--expected-documents", type=int, required=True)
    execute.add_argument("--expected-input-tokens", type=int, required=True)
    execute.add_argument("--expected-supervised-positions", type=int, required=True)
    execute.add_argument("--warmup-steps", type=int, default=8)
    execute.add_argument("overrides", nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        print(json.dumps(prepare_data(args.source, args.output, args.manifest), indent=2))
    elif args.command == "prepare-initial":
        print(json.dumps(prepare_initial(args.source, args.draft_config, args.output), indent=2))
    else:
        print(json.dumps(run(args)["metrics"], indent=2))


if __name__ == "__main__":
    main()
