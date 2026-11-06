# Source tools

Run these modules from repository root in the active ASpec venv. They are included in the source
distribution. User-facing implementations live under `aspec.commands` and ship in the wheel;
the matching `tools.*` modules remain source compatibility entry points. The wheel does not install
a top-level `tools` package.

| Task | Module |
| --- | --- |
| Create an isolated source environment, install patches and run doctor | `bash setup.sh [--dev] [--gpu]` from the repository root |
| Install into an already activated isolated venv | `bash tools/install.sh [--dev] [--gpu]` |
| Run Qwen3-8B or Ling tiny through train/resume/native export | `aspec quickstart MODEL --output NEW_PATH` |
| Run with venv-native library paths and optional private compatibility/cache directories | `bash tools/run_in_env.sh COMMAND ...` |
| Diagnose pinned runtime and GPU/native libraries | `aspec doctor [--gpu]` |
| Apply/check the current official SGLang training patch (also in the wheel) | `aspec patch-sglang [--check]` |
| Prepare original or Target-generated training data | `aspec prepare-data`, `aspec prepare-target-data` |
| Select original answers by source ID and retokenize | `aspec select-answers` |
| Convert generation journals to typed Parquet | `aspec convert-data` |
| Evaluate real decode and export/stitch assets | `aspec evaluate`, `aspec export`, `aspec stitch-mtp` |
| List/copy/plan/train formal recipes | `aspec recipes list/copy/plan/train` |
| Build and audit a release bundle (source only) | `python -m tools.build_release --outdir NEW_DIRECTORY` |
| Verify an installed wheel in the CPU fixture (source check runner) | `python tools/check_wheel_install.py` |
| Select bounded execution topology | `tools.calibration` |
| Check maintained documentation links | `tools.check_docs` |

Use each module's `--help` plus the corresponding [user guide](../docs/README.md). Model/dataset
preparation and decode can consume substantial GPU time; they are not prerequisites for the bundled
quickstart. Source paths, data identities and output directories must be supplied explicitly.

`performance/` contains comparison adapters and profiling diagnostics; see
[benchmark protocol/tools](../docs/benchmarks/README.md). Upstream framework adapters have separate
runtime/revision requirements. `validation/` contains reusable numerical, lifecycle, recovery and
asset audit programs used by tests and maintainers. These are not normal training entry points;
individual scripts may need optional upstream references or GPUs. Run only the gates relevant to a
change and preserve their original failed/unsupported boundaries.

Generated datasets, checkpoints, logs, profiler traces, environment snapshots and superseded
deployment helpers belong outside the source tree.
