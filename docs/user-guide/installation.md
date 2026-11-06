# Installation

Use Linux x86-64 with Python >=3.11, Git and a CUDA development
toolkit containing `nvcc`. Training requires NVIDIA GPUs and a CUDA-13-compatible driver;
Quickstart uses two NVLink-connected GPUs.
Source setup is the easiest complete installation. Release bundles also provide a wheel with
the public CLI, formal recipes and Quickstart data; developer validation tools remain in the source.

On Python 3.13, the pinned grammar dependency may build from source and needs Rust/Cargo on
`PATH` when no matching wheel is available; see [native prerequisites](installation-advanced.md#requirements).

## One-command setup

```bash
bash setup.sh
source .venv/bin/activate
```

Setup creates an isolated `.venv`, installs the pinned runtime, applies the SGLang training patch,
and then runs CPU doctor. The ASpec-owned DSpark Q/K CuTeDSL kernel ships in the package. Setup checks the host's glibc, C++
toolchain, CUDA compiler and RDMA libraries before downloading the runtime and gives repair guidance when a native
prerequisite is missing. It does not change system packages/drivers, download models or start Ray.

Choose a different environment location or install contributor tools when needed:

```bash
bash setup.sh --venv /local/aspec-venv --python python3.12 --dev
source /local/aspec-venv/bin/activate
```

Rerunning setup is supported for an environment created by the same checkout. Existing unowned venvs
are rejected. The Q/K kernel is selected only for compatible CUDA shapes and otherwise preserves the
eager path. `--gpu` additionally runs GPU doctor;
select an allocated idle device with `CUDA_VISIBLE_DEVICES` before using that option.

`aspec` commands prepare this venv's native-library paths automatically. The
`tools/run_in_env.sh` wrapper remains available for arbitrary Python/torchrun commands.

## First run

Choose either model; both use bundled conversations and the same two-GPU train/resume/export flow:

```bash
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart ling-tiny --output outputs/ling-tiny-001
# Or:
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart qwen3-8b --output outputs/qwen3-8b-001
```

Use `--target /absolute/path/to/model` to reuse a compatible local checkpoint. Without it, the selected
model's public Hub ID is used. Continue with [Quickstart](quickstart.md) for Plan, results and assets.

## Other environments

- [Container setup](containers.md) supplies a CUDA development base, Python, C++ and RDMA libraries;
  it runs the same source setup and still uses the host's NVIDIA driver.
- [Advanced installation](installation-advanced.md) covers native prerequisites, exact version pins,
  dependency override disclosures, private CUDA compatibility libraries, caches and manual setup.
- [Troubleshooting](troubleshooting.md) explains runtime/transport failures.

The pinned runtime is Torch 2.13.0 / SGLang 0.5.21 / CUDA 13. Doctor reports the
[two dependency metadata overrides](installation-advanced.md#dedicated-environment) explicitly.
CPU checks do not exercise GPU capacity or NVLink.

## Install a release wheel

Use the wheel and `runtime/` requirements from the same audited release bundle. Prepare the
[native prerequisites](installation-advanced.md#requirements), then install into a fresh venv:

```bash
python3.12 -m venv /local/aspec-venv
source /local/aspec-venv/bin/activate
python -m pip install uv==0.13.5
python -m uv pip install --link-mode copy --python "$VIRTUAL_ENV/bin/python" \
  --override /path/to/release/runtime/cute-overrides.txt \
  -r /path/to/release/runtime/sglang-v0.5.21.txt \
  '/path/to/release/aspec-0.1.0rc1-py3-none-any.whl[fa]'
```

The versioned runtime requirements and explicit overrides are also bundled as package resources
for doctor. The wheel does not include Torch/SGLang/CUDA binaries or model weights.
Apply and verify the pinned SGLang training patch explicitly:

```bash
aspec patch-sglang
aspec patch-sglang --check
aspec doctor
```

Application is explicit: installing the wheel does not modify SGLang. Repeating the first command
verifies an already applied patch; `--check` never writes and exits nonzero if the complete patch
is absent. Wrong versions, conflicting changes and writes outside the active venv are rejected.
Recheck after reinstalling SGLang. Git is required for exact patch application and verification.

The installed CLI supports Quickstart, recipe listing/copying/planning/training, checkpoint recovery,
Report, export, Native MTP stitching, data preparation and decode evaluation outside the source tree.
For example:

```bash
aspec recipes list
aspec recipes copy dspark --output my-dspark-config
aspec quickstart ling-tiny --plan --output outputs/ling-plan
aspec export --help
aspec evaluate --help
```

Copied YAMLs refer to input assets in this installation. You can edit or replace those paths;
run/cache outputs remain relative to your working directory or explicit overrides. Use a new
copy/output directory to preserve existing work. Full docs, benchmark comparisons, fault injection,
tests and source maintenance commands remain in the source distribution.
