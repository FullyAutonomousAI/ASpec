# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_CUTE_PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels")


def _shared_cute_requirements(requirements: list[str]) -> set[str]:
    return {
        requirement
        for requirement in requirements
        if requirement.startswith(SHARED_CUTE_PACKAGES)
    }


def test_dspark_qk_uses_the_fa_cutlass_runtime_without_requiring_quack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    fa_runtime = _shared_cute_requirements(extras["fa"])
    dspark_qk_runtime = _shared_cute_requirements(extras["dspark-qk"])

    assert dspark_qk_runtime == {
        requirement
        for requirement in fa_runtime
        if requirement.startswith("nvidia-cutlass-dsl")
    }
    assert sum(
        requirement.startswith("nvidia-cutlass-dsl")
        for requirement in dspark_qk_runtime
    ) == 1
    assert any(requirement.startswith("quack-kernels") for requirement in fa_runtime)


def test_sglang_requirements_match_the_shared_cute_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = _shared_cute_requirements(
        project["project"]["optional-dependencies"]["fa"]
    )
    requirements = {
        line.strip()
        for line in (ROOT / "requirements" / "sglang-v0.5.21.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert expected <= requirements
