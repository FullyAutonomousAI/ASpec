# Engineering priorities

The [project roadmap](../../ROADMAP.md) is the single feature-status and direction list.
This page describes how contributors approach that work; [support](../support.md) defines the
current combinations and limits.

- Preserve training objectives, document boundaries, gradients and frozen Target bindings across
  optimized backends. Use numerical comparisons before interpreting performance.
- Extend reliability beyond whole worker-process loss to additional concrete failure mechanisms;
  keep shared multi-node execution and elastic membership explicit roadmap items.
- Keep checkpoint recovery, native assets and downstream formats faithful as algorithms evolve.
- Compare normal complete workflows using [benchmark methodology](../benchmarks/methodology.md).
  Maintain reusable [validation tools](../contributing/validation.md) without making historical
  experiment archives a dependency of development.
