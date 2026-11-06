# Runtime patches

The supported Target runtime uses the bundled
[SGLang v0.5.21 patch](../aspec/compat/patches/sglang-v0.5.21.patch) against official SGLang commit
`e00930c5489053f26d86b179cee0d087f846acbb`. The patch and its application tool ship in the ASpec
wheel. In the activated, isolated venv containing ASpec and `sglang==0.5.21`, run:

```bash
aspec patch-sglang
aspec patch-sglang --check
```

The first command applies the complete patch or verifies that it is already present. `--check`
is read-only and exits nonzero when the patch is absent. Wrong versions and conflicting hunks
are rejected. Neither wheel installation nor importing ASpec applies the patch automatically.
Patch applicability does not certify every installed upstream file.
The source command `python -m tools.apply_sglang_patch` remains a compatibility wrapper.

The current patch also widens global token/row offsets before strided pointer arithmetic in the
Qwen hybrid attention/GDN prefill kernels. Real long documents can exceed int32 element offsets
even when the token count itself fits int32. This is an addressing-correctness fix, not a tile,
precision or training-objective change; short inputs alone do not validate the long-input boundary.

Source terms and inherited notices remain applicable; see the patch header,
[source references](../docs/framework/REFERENCES.md) and [`licenses/`](../licenses/).
Superseded SGLang patches are not distributed and are not a supported installation lane.
