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

"""Profile one warmed native DSpark optimizer step on resident Target features.

Pass the usual ``torchspec_benchmark run`` arguments after the profiler options.
Profiled timings are diagnostic; use the unwrapped runner for throughput claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile-step", type=int, default=8,
                        help="Zero-based optimizer step to capture, after warmup.")
    options, remaining = parser.parse_known_args()
    if options.profile_step < 0:
        parser.error("--profile-step must be nonnegative")
    options.trace_dir.mkdir(parents=True, exist_ok=False)

    import torch

    from aspec.training.dspark_trainer import DSparkTrainer
    from tools.performance import torchspec_benchmark

    original = DSparkTrainer.train_from_queue
    captured = False

    def profiled_step(self, *args, **kwargs):
        nonlocal captured
        if kwargs.get("step") != options.profile_step:
            return original(self, *args, **kwargs)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profile:
            with torch.profiler.record_function("dspark_optimizer_step"):
                result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(options.trace_dir / "step.json"))
        events = profile.key_averages(group_by_input_shape=True)
        rows = [
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_us": event.cpu_time_total,
                "self_device_us": event.self_device_time_total,
                "device_us": event.device_time_total,
                "input_shapes": event.input_shapes,
            }
            for event in events
        ]
        rows.sort(key=lambda row: row["self_device_us"], reverse=True)
        (options.trace_dir / "summary.json").write_text(
            json.dumps({"profile_step": options.profile_step,
                        "scope": "Profiled diagnostic step; not formal throughput",
                        "events": rows}, indent=2) + "\n"
        )
        print(events.table(sort_by="self_device_time_total", row_limit=35), flush=True)
        captured = True
        return result

    DSparkTrainer.train_from_queue = profiled_step
    sys.argv = [sys.argv[0], *remaining]
    try:
        torchspec_benchmark.main()
        if not captured:
            raise RuntimeError("Requested profile step was not executed")
    finally:
        DSparkTrainer.train_from_queue = original


if __name__ == "__main__":
    main()
