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

"""Evaluation has a repeatable RNG stream, never the training stream."""

import copy
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def training_attention_evidence(draft):
    """Forward-only evaluation must not overwrite training backward receipts."""
    modules = list(draft.modules())
    previous = {
        module: (module.attention_observation, copy.deepcopy(module.attention_observation))
        for module in modules if hasattr(module, "attention_observation")
    }
    try:
        yield
    finally:
        for module in modules:
            if module in previous:
                original, snapshot = previous[module]
                if isinstance(original, dict):
                    original.clear()
                    original.update(snapshot)
                module.attention_observation = original
            elif isinstance(getattr(module, "attention_observation", None), dict):
                # No training observation existed before this eval forward.
                del module.attention_observation


@contextmanager
def evaluation_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
