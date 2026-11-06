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

"""Version-scoped compatibility hooks for the supported FA4/CUTLASS lane.

The module intentionally imports only the Python standard library at import time.
Callers install these hooks immediately before importing FA4 so CPU-only imports do
not acquire a CUTLASS dependency.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from functools import wraps

_LOG = logging.getLogger(__name__)
_PATCH_MARKER = "_aspec_compilation_patch"


def install_ampere_helpers_shim() -> None:
    """Provide the legacy helper module required by the pinned FA4 release.

    CUTLASS DSL 4.4.2 and later removed ``cutlass.utils.ampere_helpers`` while
    FA4 4.0.0b34 still imports its shared-memory capacity table. Existing
    modules are never replaced; missing import metadata is repaired so repeated
    discovery remains valid.
    """

    name = "cutlass.utils.ampere_helpers"
    existing = sys.modules.get(name)
    if existing is not None:
        if existing.__spec__ is None:
            existing.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return

    try:
        cutlass_utils = importlib.util.find_spec("cutlass.utils")
        ampere_helpers = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return
    if cutlass_utils is None or ampere_helpers is not None:
        return

    import cutlass.utils  # noqa: F401 -- initialize the parent before publishing its alias

    shim = types.ModuleType(name)
    shim.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    shim.SMEM_CAPACITY = {
        "sm80": (164 - 1) * 1024,
        "sm86": (100 - 1) * 1024,
        "sm89": (100 - 1) * 1024,
    }
    sys.modules[name] = shim


def _optimization_level(name: str, default: int = 3) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer from 0 through 3, got {raw!r}") from exc
    if value not in range(4):
        raise ValueError(f"{name} must be from 0 through 3, got {value}")
    return value


def configure_fa4_cutlass_compilation() -> None:
    """Enable the persistent cache and explicit compiler policy used by FA4.

    FA4 requests ``no_cache=True`` even when CUTLASS has a stable module hash.
    ASpec permits the normal disk cache for stable modules, while preserving
    upstream behavior for hashless compilations. Compiler and ptxas optimization
    levels default to 3 and can be set independently through
    ``ASPEC_FLASH_ATTN_OPT_LEVEL`` and ``ASPEC_FLASH_ATTN_PTXAS_OPT``.
    """

    try:
        from cutlass.base_dsl.cache_helpers import default_generated_ir_path
        from cutlass.base_dsl.compiler import CompileCallable
        from cutlass.base_dsl.dsl import BaseDSL
    except ImportError:
        return

    if getattr(BaseDSL, _PATCH_MARKER, False) and getattr(
        CompileCallable, _PATCH_MARKER, False
    ):
        return

    opt_level = _optimization_level("ASPEC_FLASH_ATTN_OPT_LEVEL")
    ptxas_opt = _optimization_level("ASPEC_FLASH_ATTN_PTXAS_OPT")

    if not getattr(BaseDSL, _PATCH_MARKER, False):
        original_compile_and_cache = BaseDSL.compile_and_cache

        @wraps(original_compile_and_cache)
        def compile_and_cache_with_disk(
            self,
            module,
            module_hash,
            function_name,
            pipeline,
            args_spec,
            no_cache,
            *args,
            **kwargs,
        ):
            effective_no_cache = no_cache if module_hash is None else False
            return original_compile_and_cache(
                self,
                module,
                module_hash,
                function_name,
                pipeline,
                args_spec,
                effective_no_cache,
                *args,
                **kwargs,
            )

        BaseDSL.compile_and_cache = compile_and_cache_with_disk
        setattr(BaseDSL, _PATCH_MARKER, True)
        cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR", default_generated_ir_path)
        _LOG.debug("CUTLASS DSL disk cache enabled at %s", cache_dir)

    if getattr(CompileCallable, _PATCH_MARKER, False):
        return

    original_compile = CompileCallable._compile

    @wraps(original_compile)
    def compile_with_optimization(self, func, *args, **kwargs):
        options = (kwargs.get("options", "") or "").strip()
        if "--opt-level" not in options:
            options = f"{options} --opt-level {opt_level}".strip()
        if "--ptxas-options" not in options and ptxas_opt != 3:
            options = f"{options} --ptxas-options=-O{ptxas_opt}".strip()
        kwargs["options"] = options
        return original_compile(self, func, *args, **kwargs)

    CompileCallable._compile = compile_with_optimization
    setattr(CompileCallable, _PATCH_MARKER, True)
    if opt_level != 3 or ptxas_opt != 3:
        _LOG.debug("FA4 compilation uses opt-level=%s and ptxas-opt=%s", opt_level, ptxas_opt)
