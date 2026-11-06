#!/usr/bin/env python3
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

"""Prepare the natural-enable single-node Qwen3.8-27B/4K benchmark pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    fast_repo = root / "src" / "ASpec"
    angel_repo = root / "src" / "AngelSpecBaseline"
    fast_target_gpus = args.fast_target_gpus
    fast_draft_gpus = 8 - fast_target_gpus
    angel_target_gpus = args.angel_target_gpus
    angel_draft_gpus = 8 - angel_target_gpus
    if not 0.0 < args.angel_mem_fraction_static <= 1.0:
        raise ValueError("AngelSpec mem_fraction_static must be in (0, 1]")
    for framework, target_gpus, draft_gpus in (
        ("ASpec", fast_target_gpus, fast_draft_gpus),
        ("AngelSpec", angel_target_gpus, angel_draft_gpus),
    ):
        if target_gpus < 1 or draft_gpus < 1:
            raise ValueError(f"{framework} requires at least one Target and Draft GPU")
        if args.global_batch_size % draft_gpus:
            raise ValueError(
                f"global batch size {args.global_batch_size} is not divisible by "
                f"{framework} Draft DP {draft_gpus}"
            )

    fast_accumulation_steps = args.global_batch_size // fast_draft_gpus
    angel_accumulation_steps = args.global_batch_size // angel_draft_gpus
    # AngelSpec materializes a full optimizer-step batch before consumption, so
    # aggregate source-ring capacity must cover the whole GBS.  Keep modest
    # headroom by rounding the per-producer requirement to 32 slots.
    angel_source_ring_slots = max(
        128,
        math.ceil(math.ceil(args.global_batch_size / angel_target_gpus) / 32) * 32,
    )
    fast_case = (
        f"fast-27b-4k-gbs{args.global_batch_size}-"
        f"{fast_target_gpus}t{fast_draft_gpus}d-r1"
    )
    angel_case = (
        f"angel-27b-4k-gbs{args.global_batch_size}-"
        f"{angel_target_gpus}t{angel_draft_gpus}d-r1"
    )

    fast = _load_yaml(args.fast_template)
    fast["model"].update(
        target_model_path=str(args.model.resolve()),
        native_context_length=262144,
        draft_model_config=str(
            fast_repo / "aspec/config/dspark_qwen3_8_27b_draft_config.json"
        ),
        embedding_key="model.language_model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        norm_key="model.language_model.norm.weight",
    )
    fast["dataset"].update(
        train_data_path=str(args.data.resolve()),
        eval_data_path=None,
        chat_template=None,
        shuffle_dataset=False,
    )
    fast["training"].update(
        attention_backend="cudnn_flex_attention",
        draft_context_parallel_size=1,
        micro_batch_size=1,
        draft_accumulation_steps=fast_accumulation_steps,
        max_concurrent_batches=1,
        streaming_dispatch_with_training=True,
        pipeline_dispatch_with_training=True,
        max_seq_length=4096,
        num_epochs=100,
        num_train_steps=13,
        lr_total_steps=13,
        training_num_gpus_per_node=fast_draft_gpus,
        training_num_nodes=1,
        training_node_ips=None,
        placement_strategy="inference_first",
        fsdp_strategy="REPLICATE",
        prefetch_depth=1,
        save_interval=0,
        save_per_epoch=False,
        save_final_checkpoint=False,
        dflash_packing=False,
    )
    fast["inference"].update(
        inference_num_gpus=fast_target_gpus,
        inference_num_gpus_per_engine=1,
        inference_num_gpus_per_node=8,
        inference_node_ips=None,
        inference_fetch_batch=16,
        inference_buffer_threshold=16,
        max_sample_pool_size=16,
        inference_batch_size=2,
        target_prefill_memory_optimized=True,
    )
    fast_sglang = fast["inference"].setdefault("sglang", {})
    fast_sglang.update(tp_size=1, mem_fraction_static=0.65)
    fast_sglang.setdefault("extra_args", {}).update(
        max_running_requests=2,
        max_total_tokens=8704,
        max_prefill_tokens=8704,
    )
    fast["mooncake"].update(
        data_plane="transfer_engine",
        protocol="nvlink_intra",
        feature_affinity="any",
        require_nvlink=True,
        enable_gpu_direct=True,
        allow_host_fallback=False,
        enable_hard_pin=True,
        gpu_ring_slots=16,
        source_gpu_ring_slots=16,
        destination_gpu_ring_slots=1,
        global_segment_size="32GB",
        local_buffer_size="8GB",
        host_buffer_size="1GB",
        max_seq_len=4096,
        hidden_dim=5120,
        kv_lease_ttl_s=600,
        get_retry_max_wait_seconds=600,
    )
    fast.setdefault("logging", {})["report_to"] = "none"
    fast.setdefault("debug", {}).update(
        enable_perf_metrics=True,
        performance_measurement_required=True,
        performance_warmup_steps=2,
        performance_drain_steps=1,
    )
    fast["output_dir"] = str(root / fast_case)
    fast["cache_dir"] = str(
        root / f"cache/fast-27b-4k-{fast_target_gpus}t{fast_draft_gpus}d"
    )

    angel = _load_yaml(args.angel_template)
    angel["model"].update(
        target_model_path=str(args.model.resolve()),
        draft_model_config=str(
            angel_repo / "angelspec/config/dspark_qwen3_8_27b_draft_config.json"
        ),
        embedding_key="model.language_model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        norm_key="model.language_model.norm.weight",
    )
    angel["dataset"].update(
        train_data_path=str(args.angel_data.resolve()),
        eval_data_path=None,
        dataset_format="pretokenized",
        defer_tokenization=False,
        shuffle_dataset=False,
    )
    angel["training"].update(
        attention_backend="flex_attention",
        micro_batch_size=1,
        draft_accumulation_steps=angel_accumulation_steps,
        max_concurrent_batches=1,
        max_seq_length=4096,
        num_epochs=100,
        num_train_steps=13,
        lr_total_steps=13,
        training_num_gpus_per_node=angel_draft_gpus,
        training_num_nodes=1,
        training_node_ips=None,
        placement_strategy="inference_first",
        fsdp_strategy="REPLICATE",
        prefetch_depth=1,
        save_interval=0,
        save_per_epoch=False,
        dflash_packing=False,
    )
    angel["inference"].update(
        inference_num_gpus=angel_target_gpus,
        inference_num_gpus_per_engine=1,
        inference_num_gpus_per_node=8,
        inference_node_ips=None,
        inference_fetch_batch=8,
        inference_buffer_threshold=2,
        max_sample_pool_size=512,
        inference_batch_size=2,
    )
    angel_sglang = angel["inference"].setdefault("sglang", {})
    angel_sglang.update(
        tp_size=1,
        pp_size=1,
        nnodes=1,
        mem_fraction_static=args.angel_mem_fraction_static,
    )
    angel_sglang.setdefault("extra_args", {}).update(
        max_running_requests=2,
        max_total_tokens=8704,
        max_prefill_tokens=8704,
    )
    angel["mooncake"].update(
        data_plane="transfer_engine",
        protocol="nvlink_intra",
        require_nvlink=True,
        enable_gpu_direct=True,
        allow_host_fallback=False,
        global_segment_size="32GB",
        local_buffer_size="8GB",
        host_buffer_size="1GB",
        max_seq_len=4096,
        hidden_dim=5120,
        gpu_ring_slots=angel_source_ring_slots,
        kv_lease_ttl_s=600,
        get_retry_max_wait_seconds=600,
    )
    angel.setdefault("logging", {})["report_to"] = "none"
    angel.setdefault("debug", {})["enable_perf_metrics"] = True
    angel["output_dir"] = str(root / angel_case)
    angel["cache_dir"] = str(
        root / f"cache/angel-27b-4k-{angel_target_gpus}t{angel_draft_gpus}d"
    )

    fast_path = control / f"{fast_case.removesuffix('-r1')}.yaml"
    angel_path = control / f"{angel_case.removesuffix('-r1')}.yaml"
    fast_path.write_text(yaml.safe_dump(fast, sort_keys=False), encoding="utf-8")
    angel_path.write_text(yaml.safe_dump(angel, sort_keys=False), encoding="utf-8")

    for name, config in (("ASpec", fast), ("AngelSpec", angel)):
        assert config["training"]["fsdp_strategy"] == "REPLICATE", name
        assert config["training"]["draft_accumulation_steps"] == (
            args.global_batch_size
            // config["training"]["training_num_gpus_per_node"]
        ), name
        assert (
            config["inference"]["inference_num_gpus"]
            + config["training"]["training_num_gpus_per_node"]
            == 8
        ), name
        mooncake = config["mooncake"]
        assert mooncake["data_plane"] == "transfer_engine", name
        assert mooncake["protocol"] == "nvlink_intra", name
        assert mooncake["require_nvlink"] is True, name
        assert mooncake["enable_gpu_direct"] is True, name
        assert mooncake["allow_host_fallback"] is False, name
        assert mooncake["hidden_dim"] == 5120, name
    assert angel["dataset"]["dataset_format"] == "pretokenized"

    manifest = {
        "schema_version": "aspec.single-node-refresh/v1",
        "hardware": args.hardware.upper(),
        "workload": {
            "model": "Qwen3.8-27B",
            "sequence_length": 4096,
            "global_batch_size": args.global_batch_size,
            "optimizer_steps": 13,
            "warmup_steps": 2,
            "drain_steps": 1,
        },
        "comparison": {
            "placement": {
                "aspec": (
                    f"{fast_target_gpus} TP1 Target + "
                    f"{fast_draft_gpus} REPLICATE Draft"
                ),
                "angelspec": (
                    f"{angel_target_gpus} TP1 Target + "
                    f"{angel_draft_gpus} REPLICATE Draft"
                ),
            },
            "transport": "strict NVLink",
            "aspec_attention": "cudnn_flex_attention",
            "angelspec_attention": "flex_attention",
            "extra_tuning": False,
            "core_buffer_capacity": {
                "angelspec_source_ring_slots_per_target": angel_source_ring_slots,
                "angelspec_aggregate_source_slots": (
                    angel_source_ring_slots * angel_target_gpus
                ),
                "angelspec_sglang_mem_fraction_static": (
                    args.angel_mem_fraction_static
                ),
            },
        },
        "input": {
            "aspec": {
                "path": str(args.data.resolve()),
                "sha256": _sha256(args.data),
            },
            "angelspec": {
                "path": str(args.angel_data.resolve()),
                "sha256": _sha256(args.angel_data),
                "conversion": (
                    "packed_loss_mask expanded to equivalent dense loss_mask"
                ),
            },
        },
        "configs": {
            "aspec": {"path": str(fast_path), "sha256": _sha256(fast_path)},
            "angelspec": {
                "path": str(angel_path),
                "sha256": _sha256(angel_path),
            },
        },
        "source": {
            "aspec_root": str(fast_repo),
            "angelspec_root": str(angel_repo),
            "angelspec_baseline": (
                "frozen d3412be plus audited DSpark correctness and compatibility "
                "patches; no AngelSpec Full capabilities"
            ),
        },
    }
    (control / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", choices=("h20", "h200", "b200"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fast-template", type=Path, required=True)
    parser.add_argument("--angel-template", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--angel-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--fast-target-gpus", type=int, default=4)
    parser.add_argument("--angel-target-gpus", type=int, default=4)
    parser.add_argument("--angel-mem-fraction-static", type=float, default=0.65)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
