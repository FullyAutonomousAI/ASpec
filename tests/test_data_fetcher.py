# MIT License
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from TorchSpec 6c042a87140a84d13839e341ece2c5c3ada918bc:
# tests/test_data_fetcher.py. Namespace bindings and ASpec regression coverage adapted.

"""Tests for MooncakeDataFetcher and create_mooncake_dataloader."""

import queue
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
import torch

from aspec.data.utils import pack_loss_mask, serialize_packed_loss_mask
from aspec.feature_contract import build_target_feature_contract
from aspec.training.data_fetcher import (
    MooncakeDataFetcher,
    MooncakeDataset,
    PrefetchedDataFetcher,
    TrainSample,
    _clone_fragment_before_receive_buffer_reuse,
    create_mooncake_dataloader,
)
from aspec.training.trainer import Trainer


class MockRayQueue:
    """Mock Ray Queue using stdlib queue."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def put(self, item):
        self._q.put(item)

    def get(self, block=True, timeout=None):
        return self._q.get(block=block, timeout=timeout)


def test_queue_iterator_recreation_does_not_consume_training_rng():
    loader = create_mooncake_dataloader(
        ray_queue=MockRayQueue(),
        mooncake_store=MagicMock(),
        collator=lambda value: value,
        device=torch.device("cpu"),
    )
    before = torch.get_rng_state().clone()
    iter(loader)
    iter(loader)  # Resume recreates a loader even when the old iterator persisted.
    assert torch.equal(before, torch.get_rng_state())


class MockTargetOutput:
    """Wraps a dict of tensors with a to_tensor_dict() method, like Eagle3TargetOutput."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self._tensors)


def test_clone_fragment_synchronizes_cuda_before_receive_buffer_reuse(monkeypatch):
    clone_stream = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: clone_stream)

    class FakeTensor:
        is_cuda = True
        device = torch.device("cuda:0")

        def __init__(self, name):
            self.name = name

        def clone(self):
            return FakeTensor(f"{self.name}-clone")

    cloned = _clone_fragment_before_receive_buffer_reuse(
        {"hidden_states": FakeTensor("hs"), "input_ids": FakeTensor("ids")}
    )

    assert cloned["hidden_states"].name == "hs-clone"
    assert cloned["input_ids"].name == "ids-clone"
    clone_stream.synchronize.assert_called_once_with()


class MockMooncakeStore:
    """Mock mooncake store that stores tensors or returns random ones."""

    def __init__(self, latency: float = 0.0):
        self._data: Dict[str, Dict[str, torch.Tensor]] = {}
        self._key_counter = 0
        self.latency = latency
        self.call_count = 0
        self.call_times: List[float] = []

    def put(self, tensors: Dict[str, torch.Tensor]) -> str:
        """Store tensors and return a generated key."""
        key = f"mc_{self._key_counter}"
        self._key_counter += 1
        self._data[key] = tensors
        return key

    def put_tensors(self, key: str, tensors: Dict[str, torch.Tensor]):
        """Store tensors with a provided key."""
        self._data[key] = tensors

    def _create_random_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Create a random tensor, handling both float and integer dtypes."""
        if dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.int16, torch.int8):
            return torch.randint(0, 100, shape, dtype=dtype, device=device)
        return torch.randn(shape, dtype=dtype, device=device)

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "MockTargetOutput":
        self.call_count += 1
        self.call_times.append(time.time())
        if self.latency > 0:
            time.sleep(self.latency)

        if key in self._data:
            tensors = {k: v.to(device) for k, v in self._data[key].items()}
            return MockTargetOutput(tensors)

        tensors = {
            name: self._create_random_tensor(shape, dtypes.get(name, torch.float32), device)
            for name, shape in shapes.items()
        }
        return MockTargetOutput(tensors)

    def remove_eagle3_tensors(
        self, key: str, has_last_hidden_states: bool = False, has_target: bool = True
    ):
        """Remove tensors from store (no-op for mock)."""
        self._data.pop(key, None)


def simple_collator(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack samples into batched tensors."""
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def make_sample(idx: int) -> TrainSample:
    return TrainSample(
        mooncake_key=f"sample_{idx}",
        tensor_shapes={"input_ids": (128,), "labels": (128,)},
        tensor_dtypes={"input_ids": torch.long, "labels": torch.long},
    )


class TestMooncakeDataset:
    def test_node_local_affinity_is_checked_before_transport(self):
        store = MockMooncakeStore()
        dataset = MooncakeDataset(
            MockRayQueue(),
            store,
            torch.device("cpu"),
            feature_affinity="node_local_required",
            consumer_node_ip="10.0.0.1",
        )
        sample = make_sample(0)
        sample.metadata = {
            "feature_affinity": "node_local_required",
            "source_node_ip": "10.0.0.2",
            "assigned_draft_node_ip": "10.0.0.2",
        }

        with pytest.raises(RuntimeError, match="affinity violation"):
            dataset._load_from_mooncake(sample)

        assert store.call_count == 0
        assert dataset.get_feature_contract_status()["feature_affinity"] == {
            "mode": "node_local_required",
            "consumer_node_ip": "10.0.0.1",
            "validations": 0,
            "violations": 1,
            "state": "pending_observation",
        }

    def test_node_local_affinity_accepts_exact_source_assignment_and_consumer(self):
        dataset = MooncakeDataset(
            MockRayQueue(),
            MockMooncakeStore(),
            torch.device("cpu"),
            feature_affinity="node_local_required",
            consumer_node_ip="10.0.0.1",
        )
        metadata = {
            "feature_affinity": "node_local_required",
            "source_node_ip": "10.0.0.1",
            "assigned_draft_node_ip": "10.0.0.1",
        }

        dataset._validate_node_local_affinity(metadata, "sample")

        assert dataset.get_feature_contract_status()["feature_affinity"]["state"] == (
            "validated"
        )

    def test_iterates_samples(self):
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, device, prefetch_factor=2)
        samples = list(dataset)

        assert len(samples) == 3
        assert store.call_count == 3
        for s in samples:
            assert "input_ids" in s
            assert "labels" in s
            assert s["input_ids"].shape == (1, 128)

    def test_stops_on_none_sentinel(self):
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        ray_queue.put(make_sample(0))
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, device, prefetch_factor=2)
        samples = list(dataset)

        assert len(samples) == 1

    def test_validates_received_tensors_against_plan_contract(self):
        contract = build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=2,
            target_num_hidden_layers=4,
            aux_hidden_state_layers=[1, 2],
            store_last_hidden_states=True,
        )
        store = MockMooncakeStore()
        store.put_tensors(
            "sample",
            {
                "hidden_states": torch.zeros((3, 4), dtype=torch.bfloat16),
                "input_ids": torch.zeros((3,), dtype=torch.int64),
                # Mooncake's host-buffer path derives this local CPU view from
                # input_ids; it is not an additional transported feature.
                "input_ids_cpu": torch.zeros((3,), dtype=torch.int64),
                "last_hidden_states": torch.zeros((3, 2), dtype=torch.bfloat16),
            },
        )
        sample = TrainSample(
            mooncake_key="sample",
            tensor_shapes={
                "hidden_states": (3, 4),
                "input_ids": (3,),
                "last_hidden_states": (3, 2),
            },
            tensor_dtypes={
                "hidden_states": torch.bfloat16,
                "input_ids": torch.int64,
                "last_hidden_states": torch.bfloat16,
            },
            metadata={"aspec_feature_contract_id": contract["contract_id"]},
        )
        dataset = MooncakeDataset(
            MockRayQueue(),
            store,
            torch.device("cpu"),
            feature_contract=contract,
            feature_contract_participant="trainer_0",
        )

        loaded = dataset._load_from_mooncake(sample)

        assert loaded["hidden_states"].shape == (3, 4)
        assert loaded["input_ids_cpu"].shape == (3,)
        assert dataset.get_feature_contract_status() == {
            "participant": "trainer_0",
            "role": "consumer",
            "contract_id": contract["contract_id"],
            "state": "validated",
            "binding_validations": 0,
            "metadata_validations": 1,
            "tensor_validations": 1,
            "execution_metadata": {},
            "data_admission": {"neutralized_rank_samples": 0},
        }

    def test_vllm_pp_manifest_reassembles_layers_after_commit(self):
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        input_ids = torch.tensor([10, 20, 30])
        layer_2 = torch.full((3, 2), 2, dtype=torch.bfloat16)
        layer_8 = torch.full((3, 2), 8, dtype=torch.bfloat16)
        layer_12 = torch.full((3, 2), 12, dtype=torch.bfloat16)
        for layer_id, hidden_states in (
            (2, layer_2),
            (8, layer_8),
            (12, layer_12),
        ):
            store.put_tensors(
                f"sample_layer{layer_id}",
                {
                    "hidden_states": hidden_states,
                    "input_ids": input_ids,
                },
            )

        ray_queue.put(
            TrainSample(
                mooncake_key="sample",
                tensor_shapes={
                    "hidden_states": (3, 4),
                    "input_ids": (3,),
                    "last_hidden_states": (3, 2),
                },
                tensor_dtypes={
                    "hidden_states": torch.bfloat16,
                    "input_ids": torch.int64,
                    "last_hidden_states": torch.bfloat16,
                },
                metadata={
                    "vllm_pp_complete": True,
                    "vllm_pp_layer_manifest": [
                        {
                            "layer_id": 2,
                            "mooncake_key": "sample_layer2",
                            "role": "hidden_states",
                        },
                        {
                            "layer_id": 8,
                            "mooncake_key": "sample_layer8",
                            "role": "hidden_states",
                        },
                        {
                            "layer_id": 12,
                            "mooncake_key": "sample_layer12",
                            "role": "last_hidden_states",
                        },
                    ],
                },
            )
        )
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, torch.device("cpu"))
        samples = list(dataset)

        assert len(samples) == 1
        sample = samples[0]
        assert torch.equal(sample["hidden_states"][0], torch.cat([layer_2, layer_8], dim=-1))
        assert torch.equal(sample["last_hidden_states"][0], layer_12)
        assert torch.equal(sample["input_ids"][0], input_ids)
        assert store.call_count == 3

    def test_vllm_pp_manifest_rejects_uncommitted_sample(self):
        dataset = MooncakeDataset(
            MockRayQueue(),
            MockMooncakeStore(),
            torch.device("cpu"),
        )
        sample = TrainSample(
            mooncake_key="sample",
            tensor_shapes={
                "hidden_states": (3, 2),
                "input_ids": (3,),
                "last_hidden_states": (3, 2),
            },
            metadata={
                "vllm_pp_complete": False,
                "vllm_pp_layer_manifest": [],
            },
        )

        with pytest.raises(RuntimeError, match="Incomplete vLLM PP sample"):
            dataset._load_from_mooncake(sample)

    def test_subthreshold_samples_neutralized_not_dropped(self):
        """Empty / sub-min_loss_tokens samples are kept as zero-mask micro-batches,
        not dropped — a per-rank drop would desync FSDP collectives."""

        def packed(mask):
            return serialize_packed_loss_mask(pack_loss_mask(mask))

        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        masks = {
            "sub": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.long),  # sum 2 < 4
            "valid": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),  # sum 6
            "empty": torch.zeros(8, dtype=torch.long),  # sum 0
        }
        for key, mask in masks.items():
            ray_queue.put(
                TrainSample(
                    mooncake_key=key,
                    tensor_shapes={"input_ids": (8,)},
                    tensor_dtypes={"input_ids": torch.long},
                    packed_loss_mask=packed(mask),
                )
            )
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, torch.device("cpu"), min_loss_tokens=4)
        samples = list(dataset)

        # Count preserved (lockstep): every dispatched sample yields one micro-batch.
        assert len(samples) == 3
        sub, valid, empty = samples
        assert not sub["loss_mask"].any()
        assert valid["loss_mask"].any()
        assert not empty["loss_mask"].any()
        assert dataset.get_feature_contract_status()["data_admission"] == {
            "neutralized_rank_samples": 2
        }

    def test_usp_sharded_keeps_local_zero_loss_shard_when_global_loss_exists(self):
        """A local all-zero USP shard must not be skipped independently.

        With SP=2, the first rank can own only prompt tokens while the second
        rank owns loss-bearing response tokens.  If rank 0 skips locally and
        rank 1 trains, subsequent USP collectives are ordered differently.
        """
        full_loss_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        packed_loss_mask = serialize_packed_loss_mask(pack_loss_mask(full_loss_mask))
        sample = TrainSample(
            mooncake_key="sample",
            tensor_shapes={
                "input_ids": (1, 8),
                "hidden_states": (1, 8, 2),
                "target": (1, 8, 2),
            },
            tensor_dtypes={
                "input_ids": torch.long,
                "hidden_states": torch.bfloat16,
                "target": torch.bfloat16,
            },
            packed_loss_mask=packed_loss_mask,
            metadata={"usp_sharded": True},
        )

        outputs = []
        for sp_rank in (0, 1):
            ray_queue = MockRayQueue()
            ray_queue.put(sample)
            ray_queue.put(None)

            store = MockMooncakeStore()
            store.put_tensors(
                f"sample_usp{sp_rank}",
                {
                    "input_ids": torch.arange(sp_rank * 4, sp_rank * 4 + 4).view(1, 4),
                    "hidden_states": torch.zeros(1, 4, 2, dtype=torch.bfloat16),
                    "target": torch.zeros(1, 4, 2, dtype=torch.bfloat16),
                },
            )

            dataset = MooncakeDataset(
                ray_queue,
                store,
                torch.device("cpu"),
                usp_enabled=True,
                ttt_length=0,
            )
            dataset._sp_world_size = 2
            dataset._sp_rank = sp_rank
            dataset._sp_ring_size = 1

            tensors, skipped = dataset._usp_get_sharded_item(neutralized_count=0)
            outputs.append((tensors, skipped))

        rank0_tensors, rank0_skipped = outputs[0]
        rank1_tensors, rank1_skipped = outputs[1]

        assert rank0_tensors is not None
        assert rank1_tensors is not None
        assert rank0_skipped == 0
        assert rank1_skipped == 0
        assert not rank0_tensors["loss_mask"].any()
        assert rank1_tensors["loss_mask"].any()


class TestCreateMooncakeDataloader:
    def test_default_batch_size_is_one(self):
        """Default batch_size=1 yields one sample at a time."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            prefetch_factor=2,
        )

        batches = list(dataloader)
        assert len(batches) == 4
        for batch in batches:
            assert batch["input_ids"].shape == (1, 1, 128)
            assert batch["labels"].shape == (1, 1, 128)

    def test_batch_size_batches_samples_together(self):
        """batch_size > 1 batches multiple samples together (with padding)."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=4,
        )

        batches = list(dataloader)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape == (4, 1, 128)
        assert batches[0]["labels"].shape == (4, 1, 128)

    def test_handles_incomplete_final_batch(self):
        """Incomplete final batch still yields remaining samples."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=2,
        )

        batches = list(dataloader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (2, 1, 128)
        assert batches[1]["input_ids"].shape == (1, 1, 128)


class TestMooncakeDataFetcher:
    def test_default_batch_size_one(self):
        """Default batch_size=1 yields one sample at a time."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
        )

        batches = list(fetcher)
        assert len(batches) == 3
        assert fetcher.batch_size == 1

    def test_batch_size_parameter(self):
        """batch_size parameter controls batching (= per_dp_rank_batch_size)."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=4,
        )

        batches = list(fetcher)
        assert len(batches) == 1
        assert fetcher.batch_size == 4


class TestSynchronousFetching:
    def test_fetches_samples_synchronously(self):
        """Verify samples are fetched one at a time synchronously."""
        ray_queue = MockRayQueue()
        num_samples = 4
        store = MockMooncakeStore(latency=0.01)
        device = torch.device("cpu")

        for i in range(num_samples):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            prefetch_factor=4,
        )

        batch_count = 0
        for batch in dataloader:
            batch_count += 1

        assert batch_count == num_samples
        assert store.call_count == num_samples


class TestCacheEvalSamples:
    """Tests that cache_eval_samples drains individual samples into a flat cache."""

    def test_cache_eval_samples_drains_count(self):
        """Drain 3 samples in one call, verify _eval_cache has 3 entries."""
        import itertools

        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=1,
        )

        eval_cache: list[dict] = []

        def cache_eval_samples(count):
            for sample in itertools.islice(fetcher, count):
                cpu_sample = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                }
                eval_cache.append(cpu_sample)
            return len(eval_cache)

        assert cache_eval_samples(3) == 3
        assert len(eval_cache) == 3

        for sample in eval_cache:
            assert "input_ids" in sample
            assert "labels" in sample

    def test_cache_eval_samples_incremental(self):
        """Drain samples incrementally across multiple calls."""
        import itertools

        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(5):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=1,
        )

        eval_cache: list[dict] = []

        def cache_eval_samples(count):
            for sample in itertools.islice(fetcher, count):
                cpu_sample = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                }
                eval_cache.append(cpu_sample)
            return len(eval_cache)

        assert cache_eval_samples(2) == 2
        assert cache_eval_samples(3) == 5
        assert len(eval_cache) == 5

    def test_trainer_cache_drops_terminal_eval_work_identity(self):
        trainer = SimpleNamespace(
            _eval_data_fetcher=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2]]),
                        "_aspec_work": [
                            {"work_id": "work:eval", "execution_id": "exec:eval"}
                        ],
                    }
                ]
            ),
            _eval_cache=[],
        )

        assert Trainer.cache_eval_samples(trainer, 1) == 1
        assert "_aspec_work" not in trainer._eval_cache[0]

    def test_trainer_load_sanitizes_legacy_eval_work_identity(self, tmp_path):
        torch.save(
            [
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "_aspec_work": [
                        {"work_id": "work:eval", "execution_id": "exec:eval"}
                    ],
                }
            ],
            tmp_path / "eval_rank_0.pt",
        )
        trainer = SimpleNamespace(
            cache_rank=0,
            dp_rank=0,
            _eval_cache=[],
            _wait_for_eval_cache_save=lambda: None,
        )

        assert Trainer.load_eval_cache(trainer, str(tmp_path)) == 1
        assert "_aspec_work" not in trainer._eval_cache[0]


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_packed_fetcher_reads_only_owned_document_shards_and_copies_before_reuse(monkeypatch, cp_size):
    from aspec.data.utils import DFlashPackingCollator
    from aspec.work_lifecycle import build_work_identity

    lengths = [9, 17]
    for rank in range(cp_size):
        q = MockRayQueue()
        store = MockMooncakeStore()
        releases = []
        for doc, length in enumerate(lengths):
            key = f"doc{doc}"
            ids = torch.arange(length)[None]
            owned = torch.arange(length * rank // cp_size, length * (rank + 1) // cp_size)
            shard_hidden = owned.float()[None, :, None].expand(-1, -1, 3)
            store.put_tensors(key + f"_cp{rank}", {"input_ids": ids, "hidden_states": shard_hidden})
            q.put(TrainSample(
                key, {"input_ids": (1, length), "hidden_states": (1, length, 3)},
                {"input_ids": torch.long, "hidden_states": torch.float32},
                packed_loss_mask=f"0,{length}",
                metadata={"direct_cp_sharded": True, **build_work_identity(
                    split="train", epoch=0, source_id=key,
                )},
                data_id=key, row_id=23, last_in_row=doc == len(lengths) - 1,
            ))
        q.put(None)
        dataset = MooncakeDataset(
            q, store, torch.device("cpu"), direct_cp_enabled=True,
            packing_enabled=True, max_seq_length=40, lifecycle_observer=object(),
        )
        dataset._sp_rank, dataset._sp_world_size = rank, cp_size
        monkeypatch.setattr(dataset, "_notify_release", lambda item, outcome: releases.append((item.data_id, outcome)))
        rows = list(dataset)
        assert len(rows) == 1 and len(rows[0]) == 2
        assert releases == [("doc0", "consumed"), ("doc1", "consumed")]
        assert store.call_count == 2 and not store._data
        batch = DFlashPackingCollator(
            40, pad_to_max_length=False,
            context_parallel_rank=rank, context_parallel_size=cp_size,
        )(rows[0])
        assert len(batch["_aspec_work"]) == 2
        assert batch["hidden_states"].shape[1] == batch["packed_context_positions"].shape[1]


def test_prefetch_close_waits_for_read_and_releases_unpublished_batch():
    entered = threading.Event()
    stopped = threading.Event()
    finish_read = threading.Event()
    close_done = threading.Event()
    lease = MagicMock()
    close_errors = []

    class Reader:
        def __iter__(self):
            entered.set()
            assert finish_read.wait(5)
            yield {"_mooncake_p2p_receive_leases": [lease]}

        def request_stop(self):
            stopped.set()

    fetcher = PrefetchedDataFetcher(Reader())
    iter(fetcher)
    assert entered.wait(2)

    def close():
        try:
            fetcher.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert stopped.wait(1), "close must cancel further queue reads"
        assert not close_done.is_set(), "close returned while native read was in flight"
    finally:
        finish_read.set()
        closer.join(3)
        fetcher._thread.join(3)
    assert not closer.is_alive()
    assert not fetcher._thread.is_alive()
    assert not close_errors
    lease.release.assert_called_once_with()


def test_prefetch_close_releases_queued_receive_leases():
    lease = MagicMock()
    fetcher = PrefetchedDataFetcher(())
    fetcher._queue.put({"_mooncake_p2p_receive_leases": [lease]})
    fetcher.close()
    fetcher.close()
    lease.release.assert_called_once_with()


def test_prefetch_exhaustion_does_not_read_again(monkeypatch):
    fetcher = PrefetchedDataFetcher(())
    fetcher._queue.put(fetcher._SENTINEL)
    with pytest.raises(StopIteration):
        next(fetcher)

    def unexpected_read():
        pytest.fail("an exhausted iterator would block on the empty queue")

    monkeypatch.setattr(fetcher._queue, "get", unexpected_read)
    with pytest.raises(StopIteration):
        next(fetcher)


def test_prefetch_close_fails_closed_while_native_read_cannot_stop():
    fetcher = PrefetchedDataFetcher(())
    thread = MagicMock()
    thread.is_alive.return_value = True
    fetcher._thread = thread
    with pytest.raises(RuntimeError, match="prefetch.*(stop|exit)"):
        fetcher.close()
    thread.join.assert_called_once()


@pytest.mark.parametrize("packing", [False, True])
def test_prefetch_can_close_an_idle_native_queue(packing):
    queue_entered = threading.Event()
    source = MockRayQueue()
    original_get = source.get

    def observed_get(*args, **kwargs):
        queue_entered.set()
        return original_get(*args, **kwargs)

    source.get = observed_get
    inner = MooncakeDataFetcher(
        queue=source, mooncake_store=MagicMock(), collator=lambda value: value,
        device=torch.device("cpu"), packing_enabled=packing, max_seq_length=16,
    )
    fetcher = PrefetchedDataFetcher(inner)
    iter(fetcher)
    assert queue_entered.wait(2)
    try:
        fetcher.close()
        assert not fetcher._thread.is_alive()
    finally:
        # Make the old implementation's uninterruptible queue read exit too.
        source.put(None)
        fetcher._thread.join(3)


def test_cancelable_queue_wait_retries_empty_without_ending_stream():
    source = MagicMock()
    from ray.util.queue import Empty

    source.get.side_effect = [Empty(), Empty(), "sample"]
    dataset = MooncakeDataset(source, MagicMock(), device=torch.device("cpu"))
    assert dataset._next_queue_item() == "sample"
    assert source.get.call_count == 3


def test_prefetch_device_copy_failure_releases_receive_lease(monkeypatch):
    fetcher = PrefetchedDataFetcher(())
    lease = MagicMock()
    fetcher._queue.put({"_mooncake_p2p_receive_leases": [lease]})

    def failed_copy(_batch):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(fetcher, "_to_device", failed_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        next(fetcher)
    lease.release.assert_called_once_with()
