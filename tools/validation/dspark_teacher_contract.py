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

"""Use the production teacher-normalization boundary in validation adapters.

Reuse InferenceConfig's resolver and DFlashTrainer's optional-norm behavior;
tensor shape or value statistics do not establish a feature's normalization stage.
"""

from pathlib import Path


def teacher_contract(engine, prenorm=None):
    from aspec.config.inference_config import InferenceConfig

    if engine not in ("sgl", "sglang", "hf", "vllm", "offline"):
        raise ValueError("Unknown teacher feature engine/normalization contract")
    if prenorm is not None and type(prenorm) is not bool:
        raise ValueError("Teacher pre-norm override must be a bool")
    if engine == "offline" and prenorm is None:
        raise ValueError("Offline teacher normalization contract must be explicit")
    resolved = InferenceConfig(
        inference_engine_type=engine, last_hidden_states_prenorm=prenorm,
    ).resolve_last_hidden_states_prenorm()
    return dict(engine=engine, last_hidden_states_prenorm=resolved,
                teacher_last_hidden_states="pre_final_norm" if resolved else "post_final_norm",
                additional_final_norm_calls=int(resolved))


def fixture_teacher_contract(fixture):
    from omegaconf import OmegaConf

    inference = OmegaConf.load(Path(fixture) / "fast.yaml").inference
    return teacher_contract(inference.inference_engine_type,
                            inference.get("last_hidden_states_prenorm"))


def teacher_last_hidden_states(head, value):
    """Apply the native optional final norm exactly once, without gradients."""
    contract = getattr(head, "_aspec_teacher_contract", None)
    if not isinstance(contract, dict):
        raise ValueError("Missing teacher normalization contract")
    expected = teacher_contract(contract.get("engine"), contract.get("last_hidden_states_prenorm"))
    if contract != expected or (head.norm is not None) != expected["last_hidden_states_prenorm"]:
        raise ValueError("Teacher norm load differs from resolved native contract")
    if head.norm is None:
        return value
    import torch

    with torch.no_grad():
        return head.norm(value)


def audit_native_teacher_input(head, sample, actual):
    """Observe the unmodified native trainer's actual model-boundary tensor."""
    from types import SimpleNamespace

    import torch

    from aspec.training.dflash_trainer import DFlashTrainer
    from tools.validation.dspark_nonpacking_alignment import tensor_sha

    observed = {}

    def capture(**kwargs):
        observed["last"] = kwargs["last_hidden_states"]
        return (None,) * 7

    proxy = SimpleNamespace(model=capture, dflash=SimpleNamespace(uses_target_hidden_states=True),
                            verifier_norm=head.norm, target_lm_head_weight=head.lm_head.weight)
    DFlashTrainer._forward(proxy, sample)
    if not torch.equal(observed["last"], actual):
        raise ValueError("Validation teacher input differs from unmodified native trainer")
    return tensor_sha(observed["last"])
