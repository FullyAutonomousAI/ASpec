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

import json

import pytest

from tools.validation.recipe_capacity import audit_long_run, summarize_memory


@pytest.mark.parametrize("broken", [None, "short", "empty", "cp1", "padding"])
def test_long_capacity_requires_observed_full_context(tmp_path, broken):
    training = {
        "scheduler": {
            "observed_sequence_min": 261888,
            "observed_packed_row_min": 261888,
            "observed_packed_row_max": 261888,
            "rows_dispatched": 4,
            "packed_tokens": 4 * 261888,
        },
        "execution": {"context_parallel_size": 2, "attention_backend": "fa4"},
    }
    if broken == "short":
        training["scheduler"]["observed_sequence_min"] = 128
    elif broken == "empty":
        training["scheduler"]["rows_dispatched"] = 0
    elif broken == "cp1":
        training["execution"]["context_parallel_size"] = 1
    elif broken == "padding":
        training["scheduler"]["packed_tokens"] -= 1
    (tmp_path / "evidence.json").write_text(json.dumps({"observations": {"training": training}}))
    if broken:
        with pytest.raises(ValueError, match="actual"):
            audit_long_run(tmp_path)
    else:
        assert audit_long_run(tmp_path)["tokens_per_row"] == 261888


def test_sampled_memory_requires_all_devices_and_repeated_samples(tmp_path):
    p = tmp_path / "gpu.csv"
    header = "index, memory.used [MiB], memory.total [MiB]\n"
    p.write_text(header + "0, 10 MiB, 100 MiB\n")
    with pytest.raises(ValueError, match="Missing"):
        summarize_memory(p, expected_gpus=1)
    p.write_text(p.read_text() + "0, 20 MiB, 100 MiB\n")
    assert summarize_memory(p, expected_gpus=1)["devices"]["0"]["peak_used_mib"] == 20
    with pytest.raises(ValueError, match="Missing"):
        summarize_memory(p, expected_gpus=2)
