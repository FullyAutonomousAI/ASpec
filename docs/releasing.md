# Preparing a release

Wheels contain the runtime, public CLI, model JSON assets, formal recipes, Quickstart conversations,
runtime dependency manifests and the pinned SGLang patch. Source distributions also contain full
documentation, tests, setup and maintenance tools. Canonical resource files live in their existing
source locations; `setup.py` collects them into the wheel during the build.

## Prepare

1. Select a clean, reviewed commit and set its version in `pyproject.toml`.
2. Update the changelog, user guides, supported paths and relevant examples.
3. Check source attribution, license texts and security-reporting instructions. Model and dataset
   redistribution terms are separate from the source license.
4. Before publishing a new repository, check both files and Git history for private material.
   A reviewed source export need not include development history.

## Verify

Use a dedicated Linux venv and run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md).
Build into an empty output directory:

```bash
python -m tools.build_release --outdir /absolute/path/to/new-release-dist
```

This command builds the sdist and then its wheel, audits both, collects the two runtime requirement
files under `runtime/`, and writes `SHA256SUMS`. It rejects a nonempty output directory.
`python -m build` and `python -m tools.audit_release DIST` remain available separately.

[GitHub CI](../.github/workflows/ci.yml) runs source quality checks and CPU/offline contracts on pull
requests and configured branch pushes. It does not build distributions. Packaging-related source
contracts remain in that suite so resource and audit regressions can be found before tagging.

The separate [release build workflow](../.github/workflows/build-release.yml) runs when a `v*` tag is
pushed to GitHub. Creating a local tag alone does not trigger it. The tag must exactly match `v` plus
the version in `pyproject.toml`, for example `v0.1.0rc1` for `0.1.0rc1`; a mismatch fails before
building. The workflow runs the same build command, saves the audited release bundle as the
`aspec-distributions` workflow artifact, and runs a dependent `wheel-install` job on Python 3.11.
That job checks SHA256s, installs the built wheel in a dedicated CPU venv, and executes:

```bash
python tools/check_wheel_install.py
```

The check changes to a temporary working directory and verifies installed module origins, CLI help,
recipe copying and Plan generation, both Quickstart Plans, read-only doctor, package/build identities,
and actual SGLang patch application, repetition and conflict rejection. It uses the CPU CI dependency
subset and the official SGLang wheel without its GPU dependencies. Missing full-runtime requirements
must keep doctor non-ready; this fixture is not production dependency or GPU/training acceptance.

- Include runtime/model assets, patches, requirements, guides, recipes, source tools, source
  headers and applicable license texts.
- Exclude credentials, private endpoints, local environments, datasets, weights, checkpoints,
  profiler traces and generated logs. Build from the reviewed tree, not a dirty working directory.
- Unpack the sdist into a fresh directory and check documentation and tool entry points there.
- Install the wheel in a fresh venv and run `aspec patch-sglang` from outside the source tree
  against the pinned SGLang installation. Verify application, repeat application and read-only
  `--check`; wrong versions or conflicting hunks must fail without overwriting files.
- In a fresh environment, follow installation and Quickstart through training, recovery and export.
  For runtime changes, run the relevant [GPU, numerical and fault checks](contributing/validation.md).
  A documentation-only change does not require a new all-algorithm training matrix.

Benchmark pages should contain results, methods, relevant limitations and reproduction entry points.
Historical raw records and per-artifact hash inventories are not required release assets.
Evidence records the effective installed package hash (including the bundled patch and assets),
distribution version and build provenance when available. A wheel does not borrow Git identity from
an enclosing checkout; unavailable build revision remains unknown.

## Publish

Before pushing a version tag, verify that its exact commit has passed the CI source checks and the
applicable release checks above. The tag workflow does not rerun the source test suite. After pushing,
require both `build` and `wheel-install` to pass; an uploaded artifact alone is not installation
acceptance. A failed run can be rerun for the same tag after an infrastructure issue is resolved;
source fixes require a new reviewed commit and version/tag.

The release build only saves Actions artifacts. It does not create a GitHub Release or publish to
PyPI; publication is a separate, explicitly authorized step.

Review the exact artifacts, repository destination, version/tag and release notes. Summarize checks
actually performed and any known limitations. Publish only the reviewed source and artifacts;
retain the corresponding build and validation records separately.
