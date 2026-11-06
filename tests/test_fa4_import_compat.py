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

"""Exercise the dependency-light FA4/CUTLASS compatibility boundary."""

import importlib.machinery
import importlib.util
import sys
import types

import pytest

from aspec.compat import fa4_cutlass


@pytest.fixture
def install(monkeypatch):
    for name in ("cutlass", "cutlass.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        monkeypatch.setitem(sys.modules, name, package)
    # Track this key even when initially absent, so helper-created aliases
    # cannot leak out of the fixture.
    monkeypatch.setitem(sys.modules, "cutlass.utils.ampere_helpers", None)
    monkeypatch.delitem(sys.modules, "cutlass.utils.ampere_helpers")
    return fa4_cutlass.install_ampere_helpers_shim


def test_created_alias_can_be_discovered_repeatedly(install):
    install()
    name = "cutlass.utils.ampere_helpers"
    alias = sys.modules[name]
    assert importlib.util.find_spec(name) is alias.__spec__
    assert alias.SMEM_CAPACITY["sm80"] == 163 * 1024
    install()
    assert sys.modules[name] is alias


def test_legacy_alias_metadata_is_repaired_without_replacing_contents(install, monkeypatch):
    name = "cutlass.utils.ampere_helpers"
    alias = types.ModuleType(name)
    alias.SMEM_CAPACITY = {"existing": 123}
    monkeypatch.setitem(sys.modules, name, alias)
    install()
    assert importlib.util.find_spec(name) is alias.__spec__
    assert sys.modules[name] is alias
    assert alias.SMEM_CAPACITY == {"existing": 123}


@pytest.mark.parametrize("value", ["-1", "4", "fast"])
def test_invalid_compiler_optimization_level_is_rejected(monkeypatch, value):
    monkeypatch.setenv("ASPEC_FLASH_ATTN_OPT_LEVEL", value)
    with pytest.raises(ValueError, match="ASPEC_FLASH_ATTN_OPT_LEVEL"):
        fa4_cutlass._optimization_level("ASPEC_FLASH_ATTN_OPT_LEVEL")


def test_compilation_hooks_preserve_hashless_policy_and_explicit_options(monkeypatch):
    class BaseDSL:
        def compile_and_cache(self, *args, **kwargs):
            return args, kwargs

    class CompileCallable:
        def _compile(self, func, *args, **kwargs):
            return func, args, kwargs

    modules = {
        "cutlass": types.ModuleType("cutlass"),
        "cutlass.base_dsl": types.ModuleType("cutlass.base_dsl"),
        "cutlass.base_dsl.cache_helpers": types.ModuleType("cutlass.base_dsl.cache_helpers"),
        "cutlass.base_dsl.compiler": types.ModuleType("cutlass.base_dsl.compiler"),
        "cutlass.base_dsl.dsl": types.ModuleType("cutlass.base_dsl.dsl"),
    }
    modules["cutlass"].__path__ = []
    modules["cutlass.base_dsl"].__path__ = []
    modules["cutlass.base_dsl.cache_helpers"].default_generated_ir_path = "/tmp/cute"
    modules["cutlass.base_dsl.compiler"].CompileCallable = CompileCallable
    modules["cutlass.base_dsl.dsl"].BaseDSL = BaseDSL
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ASPEC_FLASH_ATTN_OPT_LEVEL", "2")
    monkeypatch.setenv("ASPEC_FLASH_ATTN_PTXAS_OPT", "1")

    fa4_cutlass.configure_fa4_cutlass_compilation()
    cached = BaseDSL().compile_and_cache("module", "hash", "fn", "pipe", "spec", True)
    hashless = BaseDSL().compile_and_cache("module", None, "fn", "pipe", "spec", True)
    compiled = CompileCallable()._compile("fn", options="--enable-tvm-ffi")
    explicit = CompileCallable()._compile("fn", options="--opt-level 0 --ptxas-options=-O0")

    assert cached[0][5] is False
    assert hashless[0][5] is True
    assert compiled[2]["options"] == "--enable-tvm-ffi --opt-level 2 --ptxas-options=-O1"
    assert explicit[2]["options"] == "--opt-level 0 --ptxas-options=-O0"
    first_compile = CompileCallable._compile
    fa4_cutlass.configure_fa4_cutlass_compilation()
    assert CompileCallable._compile is first_compile
