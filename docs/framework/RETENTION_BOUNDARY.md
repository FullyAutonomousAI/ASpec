# Reuse and retention

ASpec retains the dependency closure needed for its runnable Target-to-Draft training path.
It is derived from upstream implementations, not a clean-room rewrite. Do not copy unrelated
backends or tooling merely to mirror an upstream directory tree.

Before changing overlapping behavior, review the pinned upstream source, tests and dependencies.
Reuse correct, applicable implementations by default. Reimplement only when the old path violates
an invariant, has a known correctness issue, adds unrelated coupling, or a substantially smaller
implementation can be verified. Record the reason and comparison in the change description.

Classify upstream material as copied, adapted, reimplemented or inspired. Copied/adapted material
keeps all applicable copyright, license and notice text, including transitive TorchSpec → AngelSpec
origin. Describe sources and main modifications in headers or adjacent documentation. Ship required
license texts; excluding historical evidence does not remove source attribution obligations.

The retained areas include controller/actor setup, feature extraction and transport, trainers/FSDP,
checkpointing, dataset rendering, model assets and selected algorithm/attention helpers. ASpec
contracts surround these implementations; they do not require one-for-one upstream classes or
permanent module names. Review current source headers, [source references](REFERENCES.md) and
[`licenses/`](../../licenses/) for actual paths and applicable terms.

This boundary is normative for the release snapshot. Git history may explain earlier choices, but
does not expand current [support](../support.md).
