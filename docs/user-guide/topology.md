# Bounded topology selection

Use this when you need to balance Target production, transfer and Draft consumption on a dedicated
homogeneous single-node GPU allocation. It searches expert-defined capability bundles, not arbitrary
algorithms or kernels. Start with a valid config and a fixed workload/global batch.

List legal candidates without starting the GPU runner:

```bash
python -m tools.calibration auto-plan --config configs/recipes/quickstart_qwen3_8b.yaml \
  --total-gpus 8 --output topology-study.json
```

Run a bounded session only when the selected host/allocation is idle:

```bash
python -m tools.calibration auto-run --config /path/to/training.yaml --total-gpus 8 \
  --output-root outputs/topology-study --budget-seconds 7200 --run-timeout-seconds 1800
```

The runner refuses an existing Ray/GPU runtime rather than cleaning another job. Each attempt has a
separate directory and owned process group. It waits for its contexts/processes to exit before reuse;
unknown residue or timeout invalidates that attempt.

## Selection semantics

Bundles specify validated attention/CP/output/transport and ordered Target-TP choices. The compiler
enumerates integer role splits and derives accumulation/ring/pool constraints, revalidating every
candidate through the normal configuration/Plan boundary.

The smallest legal TP layer is preferred. Higher TP is a capacity fallback only after static
infeasibility or clearly attributed Target initialization/token-capacity failure at the lower layer.
Trainer/transport/unknown failures do not trigger larger TP. Higher unused layers remain recorded.

Service profiles measure production with real transport/read/release, and consumer compute excluding
source waits. End-to-end rates remain backpressure-coupled diagnostics. Measured service rates rank
candidates; healthy continuous end-to-end committed-work/wall windows make the final choice. Equivalent
short-window candidates receive formal confirmation. The model is not a global-optimum guarantee.

## Evidence and resume

`automatic-topology.json` records workload/candidate identity, commands, hardware/software, failures,
ranking and selected Plan. Eligible results need finite losses/gradient norms/rates, active required
capabilities, matching Plan identity and terminal ownership with zero resident outputs/bytes.

Resume using the same output root and protocol. A leftover running attempt is reconciled only after
resources are idle and terminal Evidence can be read; failed/invalid attempts remain visible.
Measurement settings are immutable across resume. Budget, per-run timeout and exit-wait limits may
change with recorded history. Hardware cohort includes model/count/memory/capability/interconnect;
hostnames/GPU UUIDs remain audit facts. `--hardware-label` cannot bypass physical capability checks.

Automatic selection does not authorize changing training semantics or disguising a benchmark tuning
sweep. Performance claims still follow [benchmark methodology](../benchmarks/methodology.md).
