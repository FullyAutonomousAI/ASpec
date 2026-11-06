# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Additive DSpark acceptance observations, inspired by Megatron-LM's metrics.

Probability overlap is already computed by the DSpark objective. Prefix products
are formed per block before pooling, never from pooled marginal accuracies.
These fixed-trajectory training observations are not measured verifier counters.
"""

import torch


@torch.no_grad()
def acceptance_statistics(accept_rate: torch.Tensor, eval_mask: torch.Tensor) -> torch.Tensor:
    """Return [numerator/denominator, position + tau] without a host sync.

    Tau includes one Target token. An entirely masked block contributes neither
    a bonus token nor a denominator. Masks stop prefix survival at the first
    unavailable position, including a truncated document tail.
    """
    if accept_rate.ndim != 3 or accept_rate.shape != eval_mask.shape:
        raise ValueError("Acceptance and supervision must have matching [batch, blocks, positions] shapes")
    mask = eval_mask.detach().float()
    alpha = accept_rate.detach().float() * mask
    valid_blocks = mask.bool().any(dim=-1)
    tau = 1.0 + alpha.cumprod(dim=-1).sum(dim=-1)
    numerators = torch.cat((alpha.sum(dim=(0, 1)), (tau * valid_blocks).sum().reshape(1)))
    denominators = torch.cat((mask.sum(dim=(0, 1)), valid_blocks.sum().float().reshape(1)))
    return torch.stack((numerators, denominators))


def acceptance_components(statistics: torch.Tensor) -> dict:
    """Expose observations as pooled metric pairs; they are not loss terms."""
    keys = [f"accept_rate@{index}" for index in range(statistics.shape[1] - 1)]
    keys.append("tau_probabilistic")
    return {key: (statistics[0, index], statistics[1, index]) for index, key in enumerate(keys)}
