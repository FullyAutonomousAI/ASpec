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

import pytest
import torch

from aspec.controller.packing import (
    count_eligible_anchor_starts,
    pack_work_aware_rows,
)
from aspec.data.utils import DFlashPackingCollator


def _sample(length: int, execution_id: str) -> dict:
    return {
        "input_ids": torch.arange(length).unsqueeze(0),
        "loss_mask": torch.ones((1, length), dtype=torch.long),
        "hidden_states": torch.ones((1, length, 3)),
        "last_hidden_states": torch.ones((1, length, 3)),
        "_aspec_work": {"work_id": f"work:{execution_id}", "execution_id": execution_id},
    }


def test_work_aware_packing_preserves_oldest_progress_and_uses_later_work():
    result = pack_work_aware_rows(
        [7, 6, 3, 4, 2],
        [1, 1, 30, 2, 20],
        dp_size=2,
        max_seq_length=10,
    )

    assert result is not None
    assert {0, 1}.issubset(result.selected_indices)
    assert 2 in result.selected_indices
    assert all(sum([7, 6, 3, 4, 2][index] for index in row) <= 10 for row in result.rank_rows)
    assert result.packed_tokens == 18
    assert result.useful_anchor_starts == 52


def test_work_aware_packing_rejects_unrepresentable_work():
    with pytest.raises(ValueError, match="can never fit"):
        pack_work_aware_rows([11], [1], dp_size=1, max_seq_length=10)


def test_work_aware_packing_can_reserve_a_complete_final_dispatch():
    result = pack_work_aware_rows(
        [1] * 68,
        [1] * 68,
        dp_size=4,
        max_seq_length=32,
        max_selected=64,
    )

    assert result is not None
    assert len(result.selected_indices) == 64


def test_anchor_work_count_excludes_block_tail():
    assert count_eligible_anchor_starts("2,8", sample_length=10, block_size=4) == 5


def test_packing_collator_preserves_documents_positions_and_work():
    batch = DFlashPackingCollator(8)([_sample(3, "exec:a"), _sample(2, "exec:b")])

    assert batch["input_ids"].shape == (1, 8)
    assert batch["ctx_doc_ids"].tolist() == [[0, 0, 0, 1, 1, -1, -1, -1]]
    assert batch["base_position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    assert [item["execution_id"] for item in batch["_aspec_work"]] == [
        "exec:a",
        "exec:b",
    ]


def test_packing_collator_fails_closed_on_overflow():
    with pytest.raises(ValueError, match="Packed row has"):
        DFlashPackingCollator(4)([_sample(3, "exec:a"), _sample(2, "exec:b")])


@pytest.mark.parametrize("backend,expected_length", [("fa4", 5), ("cudnn_flex_attention", 5), ("flex_attention", 8)])
def test_live_trainer_selects_unpadded_packing_for_supported_backends(monkeypatch, backend, expected_length):
    from types import SimpleNamespace

    from aspec.training import trainer as module

    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "MooncakeDataFetcher", lambda **kwargs: SimpleNamespace(**kwargs))
    trainer = SimpleNamespace(
        args=SimpleNamespace(attention_backend=backend, dflash_packing=True,
                             max_seq_length=8, draft_context_parallel_size=1),
        mooncake_store=object(), dp_rank=0, assistant_header_ids=[], end_token_ids=[],
        dynamic_loss_mask=False, last_turn_loss_only=False, skip_after_header=0,
    )
    module.Trainer.set_train_queue(trainer, object())
    batch = trainer.data_fetcher.collator([_sample(3, "a"), _sample(2, "b")])
    assert batch["input_ids"].shape == (1, expected_length)
    assert batch["attention_mask"].sum() == 5
    assert batch["base_position_ids"][0, :5].tolist() == [0, 1, 2, 0, 1]


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_controller_replicates_exact_packed_rows_to_every_cp_consumer(cp_size):
    import queue
    import threading
    from collections import deque
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from aspec.controller.training_controller import AsyncTrainingController

    cls = AsyncTrainingController.__ray_metadata__.modified_class
    controller = cls.__new__(cls)
    controller.args = SimpleNamespace(max_sample_pool_size=0, dflash_block_size=2)
    controller.dp_size, controller.sp_size = 2, cp_size
    controller._pool_lock = threading.Lock()
    controller.packing_min_samples_per_dispatch = 5
    controller.packing_max_seq = 16
    samples = [
        SimpleNamespace(
            mooncake_key=f"k{i}", data_id=f"d{i}",
            tensor_shapes={"input_ids": (1, 7), "hidden_states": (1, 7, 3)},
            tensor_dtypes={"hidden_states": "float32"}, packed_loss_mask="0,7",
            metadata={"aspec_execution_id": f"e{i}"},
        ) for i in range(4)
    ]
    controller.sample_pool = deque(samples)
    controller._lifecycle = MagicMock(resident_bytes=0)
    controller._sample_bytes = {s.mooncake_key: 1 for s in samples}
    controller._row_id_counter = 0
    controller.train_queues = [queue.Queue() for _ in range(2 * cp_size)]
    for name in ("rows_dispatched", "documents_dispatched", "tokens_dispatched",
                 "capacity_tokens", "useful_anchor_starts", "max_documents_per_row"):
        setattr(controller, "_packing_" + name, 0)
    controller._packing_min_fill_ratio = None
    controller._packing_observed_row_count = 0
    controller._packing_observed_row_min = None
    controller._packing_observed_row_max = 0
    controller._training_monitor = MagicMock()
    # Packed rows own their composition; BSHD's length-grouping/DP partition
    # must never be applied on top, including with CP consumers.
    controller._partition_results = MagicMock(side_effect=AssertionError("BSHD partition used by THD"))
    controller.batch_id = 0
    assert not controller._try_dispatch_packed(optimizer_step=1, microstep=0)
    assert controller._try_dispatch_packed(
        optimizer_step=1,
        microstep=0,
        force_progress=True,
        max_documents=4,
        return_document_count=True,
    ) == 4
    assert not controller.sample_pool
    assert controller._packing_rows_dispatched == 2
    assert controller._packing_documents_dispatched == 4
    assert controller._packing_observed_row_count == 2
    assert controller._packing_observed_row_min == 14
    assert controller._packing_observed_row_max == 14
    for call in controller._lifecycle.mark_dispatched.call_args_list:
        assert call.kwargs["expected_consumers"] == cp_size
    ids = []
    for dp_rank in range(2):
        rows = [
            list(controller.train_queues[dp_rank * cp_size + rank].queue)
            for rank in range(cp_size)
        ]
        assert all(row == rows[0] for row in rows)
        assert len(rows[0]) == 2
        assert [item.last_in_row for item in rows[0]] == [False, True]
        assert len({item.row_id for item in rows[0]}) == 1
        ids.extend(item.mooncake_key for item in rows[0])
    assert set(ids) == {"k0", "k1", "k2", "k3"}
