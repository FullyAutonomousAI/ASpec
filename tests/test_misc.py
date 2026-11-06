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

from unittest import mock

from aspec.utils.misc import get_current_node_ip


def test_get_current_node_ip_prefers_live_ray_node_manager_address():
    with (
        mock.patch("aspec.utils.misc.ray.is_initialized", return_value=True),
        mock.patch(
            "aspec.utils.misc.ray.get_runtime_context"
        ) as get_runtime_context,
        mock.patch(
            "aspec.utils.misc.ray.nodes",
            return_value=[
                {
                    "Alive": True,
                    "NodeID": "current-node",
                    "NodeManagerAddress": "192.0.2.32",
                }
            ],
        ),
        mock.patch(
            "aspec.utils.misc.ray._private.services.get_node_ip_address",
            return_value="198.51.100.126",
        ),
    ):
        get_runtime_context.return_value.get_node_id.return_value = "current-node"
        assert get_current_node_ip() == "192.0.2.32"


def test_get_current_node_ip_falls_back_before_ray_initialization():
    with (
        mock.patch("aspec.utils.misc.ray.is_initialized", return_value=False),
        mock.patch(
            "aspec.utils.misc.ray._private.services.get_node_ip_address",
            return_value="[2001:db8::1]",
        ),
    ):
        assert get_current_node_ip() == "2001:db8::1"
