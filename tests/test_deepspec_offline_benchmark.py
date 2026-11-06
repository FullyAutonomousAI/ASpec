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

from tools.performance.deepspec_offline_benchmark import (
    PretokenizedCollator,
    audit,
    decode_record,
)


def test_frozen_compact_and_binary_inputs_are_identical():
    record = {"input_ids": [1, 2, 3, 4], "packed_loss_mask": "1,3", "loss_tokens": 3}
    assert decode_record(record, 4) == decode_record(
        {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]}, 4
    )


@pytest.mark.parametrize("record,cap", [
    ({"input_ids": [1, 2], "loss_mask": [1, 1]}, 1),
    ({"input_ids": [1, 2], "loss_mask": [1]}, 2),
    ({"input_ids": [1, 2], "packed_loss_mask": "1,-1,2"}, 2),
    ({"input_ids": [1, 2], "loss_mask": [0, 0]}, 2),
    ({"input_ids": [True], "loss_mask": [1]}, 2),
    ({"input_ids": [1], "loss_mask": [1], "loss_tokens": 2}, 2),
])
def test_malformed_or_oversized_records_fail_without_filtering(record, cap):
    with pytest.raises(ValueError):
        decode_record(record, cap)


def test_collator_preserves_tokens_masks_and_native_padding():
    import torch
    pytest.importorskip("deepspec")
    collate = PretokenizedCollator(None, None, 4, 14)
    result = collate([
        {"input_ids": [3, 4], "loss_mask": [0, 1]},
        {"input_ids": [5], "loss_mask": [1]},
    ])
    assert torch.equal(result["input_ids"], torch.tensor([[3, 4], [5, 0]]))
    assert result["loss_mask"].tolist() == [[0, 1], [1, 0]]
    assert result["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_audit_counts_all_documents_and_selects_real_longest(tmp_path):
    source = tmp_path / "data.jsonl"
    records = [{"input_ids": [1], "loss_mask": [1]},
               {"input_ids": [2, 3, 4], "loss_mask": [0, 1, 1]}]
    source.write_text("".join(json.dumps(x) + "\n" for x in records))
    result = audit(source, tmp_path / "audit", 4)
    assert (result["documents"], result["input_tokens"], result["supervision"]) == (2, 4, 3)
    assert json.loads((tmp_path / "audit/longest.jsonl").read_text()) == records[1]
    with pytest.raises(FileExistsError):
        audit(source, tmp_path / "audit", 4)


def test_native_cache_roundtrip_and_spawn_loader(tmp_path, monkeypatch):
    import torch
    pytest.importorskip("deepspec")
    from deepspec.data.target_cache_dataset import (
        LocalTargetCacheWriter,
        build_target_cache_manifest,
        build_target_cache_sample_bytes,
        write_target_cache_manifest,
    )

    from tools.performance.deepspec_offline_benchmark import TimedDataset, verify_cache

    cache = tmp_path / "cache"
    cache.mkdir()
    writer = LocalTargetCacheWriter(rank_dir=str(cache), max_shard_bytes=10000)
    sample = build_target_cache_sample_bytes(
        sample_id=0, input_ids=torch.tensor([2, 3]), attention_mask=torch.ones(2),
        loss_mask=torch.tensor([0, 1]), target_hidden_states=torch.zeros(2, 4),
        target_last_hidden_states=torch.zeros(2, 2),
    )
    writer.write_sample_bytes(sample)
    writer.close()
    (cache / "samples.local.idx").rename(cache / "samples.idx")
    write_target_cache_manifest(output_dir=str(cache), manifest=build_target_cache_manifest(
        num_samples=1, shards=[{"shard_id": 0, "file_name": writer.local_shard_files[0]}],
        target_layer_ids=[0, 1], hidden_size=2,
    ))
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"input_ids": [2, 3], "loss_mask": [0, 1]}) + "\n")
    assert verify_cache(source, cache, 4)["exact_token_and_loss_mask_match"]
    monkeypatch.setenv("ASPEC_DEEPSPEC_READ_LOG", str(tmp_path))
    monkeypatch.setenv("ASPEC_DEEPSPEC_READ_RANK", "0")
    loader = torch.utils.data.DataLoader(TimedDataset(str(cache)), batch_size=1, num_workers=1,
                                        multiprocessing_context="spawn")
    assert next(iter(loader))["input_ids"].tolist() == [[2, 3]]
    source.write_text(json.dumps({"input_ids": [2, 4], "loss_mask": [0, 1]}) + "\n")
    with pytest.raises(ValueError, match="token/mask differs"):
        verify_cache(source, cache, 4)
