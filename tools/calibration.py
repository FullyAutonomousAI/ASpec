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

"""Plan, run, or inspect expert-bounded topology calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from aspec.calibration import (
    CalibrationSearchSpace,
    build_calibration_report,
    compile_calibration_plan,
)
from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan
from aspec.topology import (
    TopologyPolicy,
    compile_topology_study,
    current_capability_bundle,
    first_class_capability_bundle,
)
from aspec.topology_runner import AutomaticRunPolicy, AutomaticTopologyRunner


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _non_negative_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def _write(payload: dict, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="compile an expert-bounded candidate set")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--override", action="append", default=[])
    plan.add_argument("--total-gpus", type=int, required=True)
    plan.add_argument("--trainer-gpus", type=_positive_ints, required=True)
    plan.add_argument("--micro-batches", type=_positive_ints, default=(1,))
    plan.add_argument("--inference-batches", type=_positive_ints, default=(1,))
    plan.add_argument("--concurrency", type=_positive_ints, default=(1,))
    plan.add_argument("--pool-headroom", type=_non_negative_ints, default=(0,))
    plan.add_argument("--output", type=Path)

    collect = subparsers.add_parser("collect", help="select from completed Evidence")
    collect.add_argument("--evidence", type=Path, nargs="+", required=True)
    collect.add_argument("--equivalence-fraction", type=float, default=0.02)
    collect.add_argument("--output", type=Path)

    auto_plan = subparsers.add_parser(
        "auto-plan",
        help="derive legal role probes and all integer allocations without manual split lists",
    )
    auto_plan.add_argument("--config", type=Path, required=True)
    auto_plan.add_argument("--override", action="append", default=[])
    auto_plan.add_argument("--total-gpus", type=int, required=True)
    auto_plan.add_argument(
        "--bundle",
        action="append",
        default=[],
        help=(
            "expert capability bundle: current (default), flex_cp1_buffered_nvlink, "
            "fa4_cp1_direct_nvlink, or fa4_cp2_direct_nvlink"
        ),
    )
    auto_plan.add_argument("--buffer-headroom", type=_non_negative_ints, default=(0, 1))
    auto_plan.add_argument(
        "--target-tp-sizes",
        type=_positive_ints,
        help=(
            "ordered expert-approved Target TP fallback ladder "
            "(default: 1,2,4)"
        ),
    )
    auto_plan.add_argument("--profile-headroom", type=int, default=1)
    auto_plan.add_argument("--max-finalist-shapes", type=int, default=2)
    auto_plan.add_argument("--producer-headroom-fraction", type=float, default=0.05)
    auto_plan.add_argument("--output", type=Path)

    auto_run = subparsers.add_parser(
        "auto-run",
        help="run role probes, service-rate ranking, short finalists, and confirmation",
    )
    auto_run.add_argument("--config", type=Path, required=True)
    auto_run.add_argument("--total-gpus", type=int, required=True)
    auto_run.add_argument("--output-root", type=Path, required=True)
    auto_run.add_argument("--bundle", action="append", default=[])
    auto_run.add_argument("--buffer-headroom", type=_non_negative_ints, default=(0, 1))
    auto_run.add_argument(
        "--target-tp-sizes",
        type=_positive_ints,
        help=(
            "ordered expert-approved Target TP fallback ladder "
            "(default: 1,2,4)"
        ),
    )
    auto_run.add_argument("--profile-headroom", type=int, default=1)
    auto_run.add_argument("--max-finalist-shapes", type=int, default=2)
    auto_run.add_argument("--producer-headroom-fraction", type=float, default=0.05)
    auto_run.add_argument("--budget-seconds", type=float, default=7200.0)
    auto_run.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    auto_run.add_argument("--confirm-repeats", type=int, default=1)
    auto_run.add_argument("--equivalence-fraction", type=float, default=0.02)
    auto_run.add_argument("--calibration-steps", type=int, default=3)
    auto_run.add_argument("--calibration-warmup-steps", type=int, default=1)
    auto_run.add_argument("--calibration-drain-steps", type=int, default=0)
    auto_run.add_argument("--producer-profile-samples", type=int, default=64)
    auto_run.add_argument(
        "--producer-profile-warmup-samples", type=int, default=16
    )
    auto_run.add_argument(
        "--post-run-quiescence-timeout-seconds", type=float, default=60.0
    )
    auto_run.add_argument("--hardware-label")
    return parser


def _resolve_bundles(names, execution_plan, target_tp_sizes=None):
    names = names or ["current"]
    result = []
    for name in names:
        if name == "current":
            result.append(
                current_capability_bundle(
                    execution_plan,
                    target_tensor_parallel_sizes=target_tp_sizes,
                )
            )
            continue
        bundle = first_class_capability_bundle(
            name,
            request_batch_size=execution_plan.policy.pipeline[
                "request_batch_size"
            ],
            max_concurrent_batches=execution_plan.policy.pipeline[
                "max_concurrent_batches"
            ],
        )
        result.append(
            replace(
                bundle,
                target_tensor_parallel_sizes=(
                    target_tp_sizes or bundle.target_tensor_parallel_sizes
                ),
            )
        )
    return tuple(result)


def _topology_policy(args) -> TopologyPolicy:
    return TopologyPolicy(
        buffer_headroom_dispatches=args.buffer_headroom,
        profile_headroom_dispatches=args.profile_headroom,
        max_finalist_shapes=args.max_finalist_shapes,
        producer_headroom_fraction=args.producer_headroom_fraction,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        config = load_config(str(args.config), cli_args=args.override)
        execution_plan = compile_execution_plan(
            config, config_path=str(args.config), cli_args=args.override
        )
        payload = compile_calibration_plan(
            execution_plan,
            total_gpus=args.total_gpus,
            search_space=CalibrationSearchSpace(
                trainer_gpu_counts=args.trainer_gpus,
                micro_batch_sizes=args.micro_batches,
                inference_batch_sizes=args.inference_batches,
                max_concurrent_batches=args.concurrency,
                pool_headroom_steps=args.pool_headroom,
            ),
        ).to_dict()
    elif args.command == "collect":
        payload = build_calibration_report(
            args.evidence, equivalence_fraction=args.equivalence_fraction
        )
    elif args.command == "auto-plan":
        config = load_config(str(args.config), cli_args=args.override)
        execution_plan = compile_execution_plan(
            config, config_path=str(args.config), cli_args=args.override
        )
        payload = compile_topology_study(
            config,
            execution_plan,
            total_gpus=args.total_gpus,
            bundles=_resolve_bundles(
                args.bundle, execution_plan, args.target_tp_sizes
            ),
            policy=_topology_policy(args),
            config_path=str(args.config),
        ).to_dict()
    else:
        base_config = load_config(str(args.config))
        base_plan = compile_execution_plan(
            base_config,
            config_path=str(args.config),
        )
        runner = AutomaticTopologyRunner(
            config_path=args.config,
            output_root=args.output_root,
            total_gpus=args.total_gpus,
            bundles=_resolve_bundles(args.bundle, base_plan, args.target_tp_sizes),
            topology_policy=_topology_policy(args),
            run_policy=AutomaticRunPolicy(
                total_budget_seconds=args.budget_seconds,
                per_run_timeout_seconds=args.run_timeout_seconds,
                confirm_repeats=args.confirm_repeats,
                equivalence_fraction=args.equivalence_fraction,
                calibration_steps=args.calibration_steps,
                calibration_warmup_steps=args.calibration_warmup_steps,
                calibration_drain_steps=args.calibration_drain_steps,
                producer_profile_samples=args.producer_profile_samples,
                producer_profile_warmup_samples=(
                    args.producer_profile_warmup_samples
                ),
                post_run_quiescence_timeout_seconds=(
                    args.post_run_quiescence_timeout_seconds
                ),
            ),
            hardware_label=args.hardware_label,
        )
        payload = runner.run()
        args.output = args.output_root / "automatic-topology.json"
    _write(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
