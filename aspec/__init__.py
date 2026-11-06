# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ASpec - Torch-native speculative decoding training framework."""

from importlib import import_module

_EXPORTS = {
    "DFlashModel": "aspec.models.dflash",
    "DFlash2Model": "aspec.models.dflash2",
    "AutoDraftModelConfig": "aspec.models.draft",
    "AutoEagle3DraftModel": "aspec.models.draft",
    "DFlashConfig": "aspec.models.draft.dflash",
    "DFlashDraftModel": "aspec.models.draft.dflash",
    "DFlash2Config": "aspec.models.draft.dflash2",
    "DFlash2DraftModel": "aspec.models.draft.dflash2",
    "DSparkConfig": "aspec.models.draft.dspark",
    "DSparkDraftModel": "aspec.models.draft.dspark",
    "DSparkModel": "aspec.models.dspark",
    "Eagle3Model": "aspec.models.eagle3"
}

__all__ = [
    "Eagle3Model",
    "DFlashModel",
    "DFlashConfig",
    "DFlashDraftModel",
    "DFlash2Model",
    "DFlash2Config",
    "DFlash2DraftModel",
    "DSparkModel",
    "DSparkConfig",
    "DSparkDraftModel",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]


def __getattr__(name):
    """Defer model imports until the CLI has configured native libraries."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
