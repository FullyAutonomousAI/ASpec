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

"""Small canonical helpers shared by Plan and node-local runtime scheduling."""


def allocate_node_ready_watermarks(
    total_watermark: int,
    demand_by_node: dict[str, int],
) -> dict[str, int]:
    """Split a configured ready watermark into deterministic locality credits."""

    if total_watermark < 0:
        raise ValueError("ready watermark must be non-negative")
    if not demand_by_node or any(demand < 1 for demand in demand_by_node.values()):
        raise ValueError("node-local dispatch demand must be positive")
    if total_watermark == 0:
        return {node: 0 for node in demand_by_node}
    if total_watermark < len(demand_by_node):
        raise ValueError(
            "a bounded node-local ready watermark needs at least one credit per node: "
            f"watermark={total_watermark}, nodes={len(demand_by_node)}"
        )

    total_demand = sum(demand_by_node.values())
    credits = {node: 1 for node in demand_by_node}
    remaining = total_watermark - len(credits)
    if remaining == 0:
        return credits

    exact = {
        node: remaining * demand / total_demand
        for node, demand in demand_by_node.items()
    }
    for node, share in exact.items():
        credits[node] += int(share)
    remainder = total_watermark - sum(credits.values())
    for node in sorted(
        demand_by_node,
        key=lambda item: (-(exact[item] - int(exact[item])), item),
    )[:remainder]:
        credits[node] += 1
    return credits


def allocate_node_sample_budgets(
    total_samples: int | None,
    demand_by_node: dict[str, int],
    *,
    request_batch_size: int,
) -> dict[str, int]:
    """Reserve a complete finite producer budget for every locality domain."""

    if total_samples is None:
        return {}
    if total_samples < 0:
        raise ValueError("sample budget must be non-negative")
    if request_batch_size < 1:
        raise ValueError("request batch size must be positive")
    if not demand_by_node or any(demand < 1 for demand in demand_by_node.values()):
        raise ValueError("node-local dispatch demand must be positive")

    total_demand = sum(demand_by_node.values())
    budgets: dict[str, int] = {}
    for node, demand in demand_by_node.items():
        numerator = total_samples * demand
        if numerator % total_demand:
            raise ValueError(
                "finite node-local sample budget cannot be divided by Draft demand: "
                f"total_samples={total_samples}, demand_by_node={demand_by_node}"
            )
        budget = numerator // total_demand
        if budget % request_batch_size:
            raise ValueError(
                "finite node-local sample budget must contain complete Target requests: "
                f"node={node!r}, budget={budget}, "
                f"request_batch_size={request_batch_size}"
            )
        budgets[node] = budget
    if sum(budgets.values()) != total_samples:
        raise ValueError("node-local sample budgets do not preserve the total budget")
    return budgets
