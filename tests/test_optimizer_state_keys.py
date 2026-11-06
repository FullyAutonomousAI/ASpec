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

"""fp32 master params are checkpointed by parameter name, not by position."""

import unittest

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from aspec.training.checkpoint import OptimizerState, _fp32_param_names


class _Model(nn.Module):
    """Ordered like a draft: a body, then the head that freezing removes."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.tail_norm = nn.Parameter(torch.ones(4))


class _StubBF16Optimizer:
    """Mimics the BF16Optimizer surface OptimizerState relies on.

    The real one builds fused AdamW, which requires CUDA params, so it cannot be
    constructed in a CPU-only test.
    """

    def __init__(self, model):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.fp32_params = [p.detach().clone().float() for p in self.model_params]
        self._optim_state = {"step": 7}

    def sync_fp32_params_from_model(self):  # marks this as a BF16Optimizer to OptimizerState
        pass

    def state_dict(self):
        return self._optim_state

    def load_state_dict(self, state_dict):
        self._optim_state = state_dict


class TestFp32ParamNames(unittest.TestCase):
    def test_names_pair_with_fp32_params_positionally(self):
        """Order is whatever ``model.parameters()`` walks; the pairing just has to agree."""
        model = _Model()
        opt = _StubBF16Optimizer(model)
        by_name = dict(model.named_parameters())

        names = _fp32_param_names(opt)

        self.assertEqual(set(names), {"body.weight", "lm_head.weight", "tail_norm"})
        self.assertEqual(len(names), len(opt.fp32_params))
        for name, master in zip(names, opt.fp32_params, strict=True):
            self.assertEqual(by_name[name].shape, master.shape)

    def test_freezing_the_head_drops_only_its_name(self):
        model = _Model()
        model.lm_head.weight.requires_grad = False

        names = _fp32_param_names(_StubBF16Optimizer(model))

        self.assertEqual(set(names), {"body.weight", "tail_norm"})


class TestOptimizerStateKeying(unittest.TestCase):
    def test_state_dict_is_keyed_by_name(self):
        state = OptimizerState(_Model(), _StubBF16Optimizer(_Model()))

        keys = set(state.state_dict()["fp32_params"])

        self.assertEqual(keys, {"body.weight", "lm_head.weight", "tail_norm"})

    def test_round_trip_restores_each_master_copy(self):
        model = _Model()
        opt = _StubBF16Optimizer(model)
        saved = OptimizerState(model, opt).state_dict()
        saved = {
            "optim": saved["optim"],
            "fp32_params": {k: v.clone() for k, v in saved["fp32_params"].items()},
        }

        for master in opt.fp32_params:
            master.zero_()
        OptimizerState(model, opt).load_state_dict(saved)

        for name, master in zip(_fp32_param_names(opt), opt.fp32_params):
            torch.testing.assert_close(master, saved["fp32_params"][name])

    def test_unfreezing_a_parameter_is_rejected_rather_than_misaligned(self):
        """The two-stage case: stage 1 froze the head, stage 2 does not.

        Under positional keys, every parameter after the head would silently receive
        another parameter's saved tensor.
        """
        frozen_model = _Model()
        frozen_model.lm_head.weight.requires_grad = False
        frozen_opt = _StubBF16Optimizer(frozen_model)
        saved = OptimizerState(frozen_model, frozen_opt).state_dict()
        self.assertNotIn("lm_head.weight", saved["fp32_params"])

        thawed_model = _Model()  # head trainable again
        thawed_opt = _StubBF16Optimizer(thawed_model)

        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            OptimizerState(thawed_model, thawed_opt).load_state_dict(saved)


class _RealAdamBF16Surface:
    """Real Adam and DCP, tiny CPU tensors; no CUDA startup or fake load."""

    def __init__(self, model):
        self.model = model
        self.fp32_params = [p.detach().clone().float().requires_grad_() for p in model.parameters()]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=0.001)

    def sync_fp32_params_from_model(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, value):
        self.optimizer.load_state_dict(value)


def test_dcp_exact_resume_restores_lazy_adam_moments_without_an_extra_update(tmp_path):
    original = _RealAdamBF16Surface(nn.Linear(3, 2))
    original.fp32_params[0].grad = torch.ones_like(original.fp32_params[0])
    original.optimizer.step()
    # The unused bias is allowed to remain lazy, even when an auditor touched it.
    original.optimizer.state[original.fp32_params[1]]
    destination = tmp_path / "optimizer"
    dcp.save({"optim_state": OptimizerState(original.model, original)}, checkpoint_id=destination)
    resumed = _RealAdamBF16Surface(nn.Linear(3, 2))
    dcp.load(
        {"optim_state": OptimizerState(resumed.model, resumed, load_path=destination)},
        checkpoint_id=destination,
    )
    for reference, actual in zip(original.fp32_params, resumed.fp32_params, strict=True):
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)
        first, second = original.optimizer.state[reference], resumed.optimizer.state[actual]
        assert first.keys() == second.keys()
        for field in first:
            torch.testing.assert_close(first[field], second[field], rtol=0, atol=0)
    for optimizer in (original, resumed):
        optimizer.fp32_params[0].grad = torch.full_like(optimizer.fp32_params[0], 0.25)
        optimizer.optimizer.step()
    torch.testing.assert_close(original.fp32_params[0], resumed.fp32_params[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
