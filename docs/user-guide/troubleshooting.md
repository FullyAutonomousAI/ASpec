# Troubleshooting

Start with `aspec doctor --gpu`, and the failed run's
`aspec --report RUN --report-format json`. Preserve the original error/Evidence before retrying.

| Symptom | Action |
| --- | --- |
| Wrong/missing command or imports | Activate the intended venv; inspect `python -m pip show aspec`; source tools run from repository root |
| Resolver conflict | Use pinned requirements in a fresh venv; no `--no-deps` or wheel metadata edits |
| Missing training hook | Run patch helper `--check`; wrong versions/unknown changes require a clean install |
| CUDA/driver failure | Compare Torch runtime, driver and private compatibility paths; follow installation |
| `libcudart.so.12` missing | Put this venv's `nvidia/cuda_runtime/lib` in `LD_LIBRARY_PATH`; CUDA13 does not provide the CUDA12 SONAME |
| Strict NVLink/IPC fails | Check peer topology and container permissions; keep the requested transport enabled when diagnosing the failure |
| `No RDMA devices found` during an NVLink run | Topology discovery can emit this warning without an HCA. Check the selected transport and Evidence; it does not by itself mean NVLink failed |
| Slow first step | Allow model loading, cold imports and JIT; inspect job activity, do not restart unknown-state runs |
| OOM or full ring/pool | Inspect context/capture bytes and ready/in-flight/source/destination/host capacities separately |
| No usable samples | Check template/key/assistant mask/minimum supervision and admission counts |
| No complete eval DP batch | Supply enough admitted held-out rows; a DP-aligned tail can be truncated |
| Managed output conflict | Use a fresh output; resume reads the old checkpoint but writes a new run |
| Resume mismatch | Restore the same semantics or explicitly start a weights-only phase; never edit checkpoint metadata |
| Reader fence or prefetch close failure | Preserve cleanup Evidence and identify this run's remaining reader processes; do not manually release registered source buffers while those readers may be active |
| Invalid learning rate, clipping or timeout | These values must be finite and positive; NaN/Inf do not disable validation or deadlines |
| `incomplete_evidence` | Read pending capabilities; a configured option is not observed proof |
| Decode saves JSON but exits `1` | Inspect verdict, every repeat, lengths/cache metadata; see current support limitation |
| `resource_tracker` restarted between evaluation engines | Can occur during SGLang shutdown. Save the warning and inspect leftover processes and OS shared-memory resources before starting another run |
| Unsupported export | Use documented native format or report unsupported; do not drop indexer/compressor/mapping tensors |

## Reproducible issues

Include revision/dirty state, install commands, Python/package versions, GPU/driver, doctor output,
command/config, error, Plan/Evidence and minimal non-sensitive input. Separate requested settings
from observed facts. Redact tokens, tracking credentials, private prompts/paths and Ray auth files.
Do not upload checkpoints, weights or datasets by default. Use the repository's issue channel when
available; security-sensitive reports follow [SECURITY.md](../../SECURITY.md).

## Backend startup errors

SGLang actor and child-process output is retained under the run directory in `backend_logs/sglang-*.log`. If a run fails, Evidence and `aspec --report` include bounded log tails alongside the original driver exception. This preserves startup details such as `EADDRINUSE` even when Ray reports only an actor death. Each excerpt is limited to the last 8 KiB; full logs remain in the run directory.

The pinned SGLang lane intentionally overrides two upstream CuTe requirements. `pip check`
reports those metadata mismatches; follow the explicit override installation in the
[installation guide](installation.md). Doctor reports those exact exceptions separately and
rejects unrelated dependency conflicts. Do not use `--no-deps` or modify wheel metadata to
make an unexplained mismatch disappear.
