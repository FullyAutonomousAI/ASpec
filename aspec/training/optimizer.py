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

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from aspec.training.lr_scheduler import LRSchedulerWithWarmup
from aspec.utils.logging import print_on_rank0


def _copy_tensors_(destinations, sources):
    has_dtensor = any(isinstance(tensor, DTensor) for tensor in destinations) or any(
        isinstance(tensor, DTensor) for tensor in sources
    )
    if has_dtensor:
        for destination, source in zip(destinations, sources, strict=True):
            destination.copy_(source)
        return
    torch._foreach_copy_(destinations, sources)


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        min_lr=0.0,
        wsd_decay_steps=None,
        wsd_decay_style=None,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self._fuse_clip_scaling = (
            bool(self.model_params)
            and max_grad_norm > 0
            and all(type(p) is torch.nn.Parameter and p.is_cuda for p in self.model_params)
            and torch.cuda.get_device_capability(self.model_params[0].device) == (9, 0)
        )
        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        if not getattr(self.optimizer, "_step_supports_amp_scaling", False):
            raise RuntimeError(
                "BF16Optimizer requires fused AdamW with device-side found_inf support"
            )
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
            decay_style=decay_style,
            min_lr=min_lr,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            grad_destinations = []
            grad_sources = []
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    grad_destinations.append(g)
                    grad_sources.append(p.grad)
                    mp.grad = g
                else:
                    mp.grad = None
            if grad_destinations:
                _copy_tensors_(grad_destinations, grad_sources)

        if self._fuse_clip_scaling:
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in self.fp32_params if p.grad is not None]
            )
            # Fused AdamW already supports device-side gradient unscaling.
            # Apply the same clipping coefficient there, avoiding a complete
            # read/write pass over the FP32 gradient buffers on Hopper.
            coefficient = (self.max_grad_norm / (grad_norm + 1e-6)).clamp(max=1.0)
            self.optimizer.grad_scale = coefficient.reciprocal()
        else:
            self.optimizer.grad_scale = None
            grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)

        # Fused AdamW consumes this device scalar through the same path used by
        # GradScaler. A nonzero value skips parameter, moment, weight-decay, and
        # optimizer-step updates without synchronizing the CUDA scalar to Python.
        found_inf = (~torch.isfinite(grad_norm)).to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # A sharded rank may be the only one that observes a nonfinite
            # gradient. All training ranks must make the same update decision.
            collective_found_inf = (
                found_inf.to_local() if isinstance(found_inf, DTensor) else found_inf
            )
            dist.all_reduce(collective_found_inf, op=dist.ReduceOp.MAX)
        self.optimizer.found_inf = found_inf
        self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            _copy_tensors_(self.model_params, self.fp32_params)
            for p in self.model_params:
                p.grad = None

        # DTensor norms may still carry a Partial norm-reduction placement.
        # item() on that value reports only this rank's local contribution.
        # Keep clipping unchanged, but return the observable global norm.
        return grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            _copy_tensors_(self.fp32_params, self.model_params)

    def state_dict(self):
        return self.optimizer.state_dict()

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
