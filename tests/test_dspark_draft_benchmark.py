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

"""CPU-only checks for the bounded Draft performance workload."""

from types import SimpleNamespace

import pytest

from tools.performance.dspark_draft_benchmark import configure_target_context, document_lengths


@pytest.mark.parametrize("capacity", [4096, 131072])
def test_single_document_is_full(capacity):
    assert document_lengths(capacity, "single") == [capacity]


@pytest.mark.parametrize("capacity", [4096, 131072])
def test_packed_workload_contains_unequal_whole_documents(capacity):
    lengths = document_lengths(capacity, "packed")
    assert len(lengths) == len(set(lengths)) == 4
    assert min(lengths) > 7
    assert sum(lengths) <= capacity
    assert 0.94 <= sum(lengths) / capacity <= 0.96


def test_long_target_preserves_original_rope_theta():
    config = SimpleNamespace(
        rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
        max_position_embeddings=40960,
    )
    configure_target_context(config, 131072)
    assert config.rope_parameters["rope_theta"] == 1000000
    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 8.0
    assert config.max_position_embeddings == 131072


def test_short_target_configuration_is_unchanged():
    config = SimpleNamespace(
        rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
        max_position_embeddings=40960,
    )
    configure_target_context(config, 4096)
    assert config.rope_parameters == {"rope_theta": 1000000, "rope_type": "default"}
    assert config.max_position_embeddings == 40960
