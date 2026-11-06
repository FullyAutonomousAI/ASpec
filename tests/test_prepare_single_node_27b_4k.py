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

from argparse import Namespace
from pathlib import Path

import yaml

from tools.performance.prepare_single_node_27b_4k import prepare


def _write_template(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {},
                "dataset": {},
                "training": {},
                "inference": {"sglang": {}},
                "mooncake": {},
            }
        ),
        encoding="utf-8",
    )


def test_prepare_marks_angelspec_input_as_pretokenized(tmp_path: Path) -> None:
    fast_template = tmp_path / "fast.yaml"
    angel_template = tmp_path / "angel.yaml"
    _write_template(fast_template)
    _write_template(angel_template)
    fast_data = tmp_path / "fast.parquet"
    angel_data = tmp_path / "angel.parquet"
    fast_data.write_bytes(b"fast-input")
    angel_data.write_bytes(b"angel-input")
    model = tmp_path / "model"
    model.mkdir()
    root = tmp_path / "benchmark"

    prepare(
        Namespace(
            hardware="h20",
            root=root,
            fast_template=fast_template,
            angel_template=angel_template,
            data=fast_data,
            angel_data=angel_data,
            model=model,
            global_batch_size=510,
            fast_target_gpus=6,
            angel_target_gpus=5,
            angel_mem_fraction_static=0.65,
        )
    )

    config = yaml.safe_load(
        (root / "control/angel-27b-4k-gbs510-5t3d.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["dataset"]["dataset_format"] == "pretokenized"
    assert config["dataset"]["defer_tokenization"] is False
    assert config["dataset"]["train_data_path"] == str(angel_data.resolve())
