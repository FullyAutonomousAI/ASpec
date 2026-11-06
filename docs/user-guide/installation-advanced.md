# Advanced installation and native prerequisites

Start with [one-command source setup](installation.md). This page explains its pinned lane,
native prerequisites, compatibility exceptions and manual recovery procedures.

The [release wheel](installation.md#install-a-release-wheel) includes public commands, formal recipes,
Quickstart data and the pinned patch. Source setup and developer `python -m tools.*` commands below
run from the repository root. Both installation forms use the same runtime lane and overrides.

## Requirements

- Linux x86-64 with **glibc 2.34 or newer** (required by the pinned SGLang wheel); Python >=3.11. Check the host with `getconf GNU_LIBC_VERSION`.
- A CUDA-13-compatible NVIDIA driver/runtime. This lane pins Torch2.13.0, SGLang0.5.21,
  Transformers5.12.1, FA4 4.0.0b34, CUTLASS DSL 4.7.1 and Quack 0.6.5.
- Git, a C++20 compiler **and standard library supporting `std::bit_cast`**, and sufficient disk/host memory for CUDA wheels, models and checkpoints. The default SGLang JIT requires both compiler and library support; the presence of `g++` alone is insufficient.
- The quickstart requires **two NVLink-connected GPUs**; larger recipes differ.

Direct pins are in [requirements/sglang-v0.5.21.txt](../../requirements/sglang-v0.5.21.txt). They
start from the [official pinned SGLang manifest](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/pyproject.toml)
and apply the ASpec-validated CUTLASS DSL 4.7.1 / Quack 0.6.5 compatibility override.
CPU planning does not imply CPU training support.

On Python 3.13, `outlines-core==0.1.26`, inherited through the pinned SGLang/Outlines dependency,
may require a source build. Install a Rust toolchain (`rustc` and `cargo`) on `PATH` when no
matching wheel is available; uv builds the extension during setup. Setup does not install host
toolchains. The runtime lane pins `nvidia-cudnn-frontend==1.31.0` to retain CUTLASS DSL 4.7.1
compatibility; frontend 1.32 requires CUTLASS >=4.8.

The C++20 requirement comes from the pinned SGLang JIT: its
[tensor matcher](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/kernels/jit/include/sgl_kernel/tensor.h#L595)
uses `std::bit_cast`, and its JIT compilation flags select C++20. Setup performs this check
automatically; it is not a separate manual Quickstart step.

The Mooncake native extension also requires the host RDMA libraries `libibverbs.so.1` and `libmlx5.so.1` (typically supplied by rdma-core; package names vary by distribution). These are native prerequisites, separate from the Python wheels. Provision them in the base image or through the system administrator, and ensure they are visible to the dynamic loader in every worker process.

Check the C++ compiler and standard library used by the JIT before starting GPU services:

```bash
printf '#include <bit>\nint main() { return std::bit_cast<unsigned>(1.0f) == 0; }\n' | "${CXX:-c++}" -std=c++20 -x c++ -fsyntax-only -
```

SGLang also invokes the CUDA toolkit's `nvcc` for JIT kernels. Setup checks the executable selected
by `CUDA_HOME`/`CUDA_PATH`, or `PATH` when those variables are unset. A CUDA-enabled PyTorch wheel
alone does not establish that this compiler is discoverable. The optional container supplies a CUDA
13.0.2 development toolkit. Setup preserves an explicit host compiler/toolkit selection.

Use `python -m tools.doctor --gpu` below to check the native Mooncake import in the selected venv. A missing RDMA library should be resolved in the environment before training.

## Dedicated environment

```bash
python3.12 -m venv ../aspec-venv
source ../aspec-venv/bin/activate
bash tools/install.sh
# Contributors use: bash tools/install.sh --dev
```

The installer checks venv isolation, Linux/Python/glibc and C++20 `std::bit_cast`, installs the
pinned requirements with their explicit uv overrides, applies/verifies the SGLang patch, then runs
CPU doctor. `--gpu` additionally probes CUDA and Mooncake; select an idle GPU first. It does not
create a venv or modify system packages/drivers. For each later command use
`bash tools/run_in_env.sh COMMAND ...`; the example launcher uses this wrapper automatically.
It discovers native libraries inside the active venv and rejects `PYTHONPATH`/`PYTHONHOME` injection.

The underlying install command, for inspecting or managing the same environment manually, is:

```bash
python -m uv pip install --link-mode copy --python "$VIRTUAL_ENV/bin/python" \
  --override requirements/cute-overrides.txt -r requirements/sglang-v0.5.21.txt -e ".[fa]"
aspec patch-sglang
```

The official SGLang 0.5.21 metadata requires CUTLASS DSL 4.6.2 and Quack 0.6.4. ASpec's
shared FA4/Q/K lane uses 4.7.1 and 0.6.5 through the explicit
[override file](../../requirements/cute-overrides.txt). Ordinary pip resolution rejects this
combination. The override leaves upstream metadata intact: `python -m pip check` reports these
two mismatches. `python -m tools.doctor` discloses the exact exceptions and fails for any other
active missing/conflicting dependency. It does not label the original metadata satisfied.
`cuda-tile==1.6.0rc5`, required by SGLang, is pinned explicitly so resolution does not require
allowing prereleases for unrelated dependencies.

CUDA dependencies occupy many gigabytes and contain many small files. Installation and cold JIT can
be slow on shared filesystems; use local SSD storage for the venv/caches when available. Models and
checkpoints can remain on shared storage. Do not restart an active installation just because its log
is quiet while extracting wheels.

The helper only modifies the pinned official wheel inside the active venv. It checks before applying,
verifies afterward, and accepts a second invocation without reapplying. Wrong versions or conflicting
patch hunks are rejected. This tests patch applicability/reversibility, not the integrity of every
SGLang file; unrelated local modifications are not certified. For an editable SGLang checkout,
review/apply the patch manually at official
commit `e00930c5489053f26d86b179cee0d087f846acbb` in an isolated checkout.
Read-only `--check` and doctor can inspect that source install without modifying it.
Both the helper and doctor reject a venv that explicitly enables `system-site-packages`.

Do not use `--no-deps`, change wheel metadata, use system-site-packages or borrow another venv's
imports to hide a failed resolver.

## DSpark Q/K specialization

ASpec ships the DSpark Q/K RMSNorm + indexed-RoPE forward and backward under
`aspec.kernels.cutedsl`, without an external cuDNN source checkout or custom build. The implementation
supports separate context/draft K planes, asymmetric GQA heads, deterministic FP32 norm-weight
reduction, and the same BF16/FP16 operand precision as the eager reference. It requires
`nvidia-cutlass-dsl[cu13]==4.7.1`; the default `fa` extra already provides that pin and retains
`quack-kernels[cu13]==0.6.5` for the other fused training operations.

Install the normal runtime, or only the kernel dependency for a focused environment:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" --override requirements/cute-overrides.txt \
  -r requirements/sglang-v0.5.21.txt -e ".[fa,dspark-qk]"
python -m tools.doctor
```

See the [kernel shape checks](../../aspec/kernels/cutedsl/dspark_qk_norm_rope.py)
and [numerical regression cases](../../tests/test_dspark_qk_cutedsl.py) for supported
geometry and comparisons against the eager reference.

Keep `flash-attn-4` inside the isolated venv and verify that it reports `4.0.0b34`.
Development wheels or unrelated system installations can have incompatible import interfaces.

ASpec's narrow FA4 compatibility adapter supplies the legacy Ampere shared-memory table required
by b34 and permits CUTLASS's persistent disk cache for compilations with a stable module hash.
`CUTE_DSL_CACHE_DIR` selects that cache location. `ASPEC_FLASH_ATTN_OPT_LEVEL` and
`ASPEC_FLASH_ATTN_PTXAS_OPT` accept integers from 0 through 3; both default to 3 for the validated
performance lane. Lower values change compilation policy and can slow kernels, so release and
benchmark runs should keep the defaults unless they explicitly study that tradeoff. Hashless
compilations retain FA4's no-cache behavior.

## Native library paths

The retained Mooncake wheel also links CUDA12 runtime, independently of Torch CUDA13. Requirements
install that library inside this venv. The wrapper sets this and Torch's library directory for the
command and its child processes, without altering the parent shell:

```bash
bash tools/run_in_env.sh python -m tools.doctor
aspec patch-sglang --check
bash tools/run_in_env.sh python -m tools.doctor --gpu
aspec --help
```

If running bare commands from other guides, first set the equivalent environment in that shell:

```bash
task_purelib=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$task_purelib/nvidia/cuda_runtime/lib:$task_purelib/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Optionally set `ASPEC_CACHE_HOME` to a dedicated local SSD directory before invoking the wrapper.
It derives absolute JIT-cache paths there. Leave it unset to retain upstream defaults. It does not
move models, checkpoints or Ray's temporary directory, and does not clean existing caches.
For an existing multi-node Ray cluster, configure node-local caches when starting each node's
Ray processes; setting driver-side caches does not reconfigure already running workers.

Doctor does not download models or start services. Without `--gpu` it checks versions, active dependency metadata and patch state;
with `--gpu` it also performs a small BF16 CUDA matmul and imports Mooncake's native extension.
Neither certifies NVLink, RDMA, training or every backend.

## Older data-center drivers

The "CUDA Version" in `nvidia-smi` is not the installed PyTorch runtime version. Supported systems
may use application-private compatibility libraries; follow the
[NVIDIA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
and administrator policy. Not every driver feature becomes available.

For an administrator-approved `cuda-compat-13-0` package, extract it without replacing the system driver:

```bash
# Obtain the appropriate NVIDIA package for your OS. This does not install a system driver.
dpkg-deb -x /path/to/cuda-compat-13-0.deb /path/to/private-compat
export ASPEC_CUDA_COMPAT_DIR=/path/to/private-compat/usr/local/cuda-13.0/compat
bash tools/run_in_env.sh python -m tools.doctor --gpu
```

Do not change global `ldconfig`, CUDA symlinks or the host driver for this quickstart. Driver and
actor processes must inherit the library paths.

## What remains explicit

| Step | Automation boundary |
| --- | --- |
| Python dependencies, two declared overrides, SGLang patch | `tools/install.sh`; patch check is idempotent |
| Venv-native library paths and optional private compatibility/cache paths | `tools/run_in_env.sh`; example launcher delegates to it |
| Base Python/venv, C++ compiler, glibc, RDMA host libraries and driver | Provision the host first; checks do not install system packages |
| Private CUDA compatibility package | Obtain/extract an appropriate NVIDIA package explicitly, then set the directory above |
| DSpark Q/K source specialization | ASpec package source plus the pinned `nvidia-cutlass-dsl[cu13]` runtime; no external source patch/build step |
| Models, datasets, upstream comparison sources | Explicit user inputs/downloads; no private checkout or developer mount required |
| GPUs, output directories and multi-node membership | Explicit run configuration; installer never starts training or a cluster |

Continue with the [quickstart](quickstart.md), or see [troubleshooting](troubleshooting.md).
