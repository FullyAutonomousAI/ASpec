# DSpark attention article figures and evidence

For [DSpark Attention: From Flexible Masks to a 3.23×
Speedup](../../dspark_attn_kernel_optimization.md).

Imported from `icavan/random-notes` revision
`84300ec9468d70207d56caa9e9684f53bedb1d7a`; article-relative paths were adapted
for the ASpec documentation tree. The artwork and evidence bytes are unchanged.

| Figure | Editable SVG | PNG |
| --- | --- | --- |
| DSpark/DFlash-style visible regions | [SVG](01-visible-regions.svg) | [PNG](01-visible-regions.png) |
| Paired five-way complete F+B comparison | [SVG](02-five-backends.svg) | [PNG](02-five-backends.png) |
| Generic mask state and forward register pressure | [SVG](03-register-pressure.svg) | [PNG](03-register-pressure.png) |

All artwork is original ASpec explanatory material, Apache-2.0. Figures 1 and 3
are schematics; their inspection windows and register-capacity bars are not
measured GPU tile dimensions or register-allocation counts. Figure 2 is a
Matplotlib plot of the archived experiment, with observed min/max window ranges.
The PNGs are 2× exports of the design canvas; SVG text remains editable.

Figure 1 shows the exact noncausal-within-draft-block mask used by this study.
It is common to the selected DSpark and DFlash-style configurations, not a claim
that their complete models or objectives are identical. Toy contexts contain 48
and 36 tokens, with five anchors total, only to make visible regions legible.
The benchmark instead has 131072 context tokens and 6144 anchors.

The five-way measurement uses the same frozen tensors, document boundaries,
anchor positions and per-document supervision budget for every route. The
[evidence JSON](benchmark-evidence.json) retains every timing window, precision
comparison, NCU counter and the exact frozen-input/source hashes. Its metadata
discloses the synthetic inputs, native sampler, package versions, architecture,
timing boundary and numerical gates. Machine-specific logs, the 1.4 GB tensor
file and full NCU reports remain in the original evidence bundle.

The two optimized paths are distinct:

- Patched FA4 is an exact interval `mask_mod` callback, retaining the original
  forward/backward sparse lists and FA4 core source.
- Patched cuDNN uses native forward and the ASpec-owned joint-GQA backward,
  including its shared-specific dS publication. The installed native wheel is
  unchanged. The source is adapted from cuDNN Frontend with retained provenance.

Figure 3's 4.37 ms is NCU **forward-main** duration, not the 21.15 ms complete
F+B graph timing. Tensor activity is not achieved useful FLOPs divided by peak,
and does not establish a speed-of-light limit. LSU local requests are not HBM
bytes or spill-buffer capacity. Native forward has zero observed local traffic
in this capture; the owned backward still has 0.189 GB. This is not a claim
that every cuDNN kernel is spill-free.

Colleagues separately validated training loss and speculative acceptance in
end-to-end runs. The evidence in this directory records the five-way kernel
benchmark; its timings cover attention F+B rather than full training throughput.
