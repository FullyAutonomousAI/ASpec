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

"""Observe DeepSpec's native cache-first workflow without replacing its math.

Run as a module from an isolated DeepSpec checkout containing this tool. The
input boundary accepts frozen token IDs, not conversations to retokenize. Cache
preparation, binary codec, distributed sampler, collator, prefetch, training,
optimizer and checkpoints are upstream implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from tools.performance.deepspec_benchmark import adapt_state, sha256


def decode_record(record, max_length):
    tokens = record.get("input_ids")
    if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("invalid input_ids")
    if len(tokens) > max_length:
        raise ValueError("input exceeds the cap; truncation is forbidden")
    mask = record.get("loss_mask")
    if mask is None:
        packed = record.get("packed_loss_mask")
        if not isinstance(packed, str) or not packed:
            raise ValueError("missing loss mask")
        segments = [int(x) for x in packed.split(",")]
        if any(x < 0 for x in segments) or sum(segments) != len(tokens):
            raise ValueError("invalid compact mask length")
        mask = [value for i, count in enumerate(segments) for value in [i % 2] * count]
    if (not isinstance(mask, list) or len(mask) != len(tokens)
            or any(type(x) is not int or x not in (0, 1) for x in mask) or not sum(mask)):
        raise ValueError("invalid or empty loss mask")
    if "loss_tokens" in record and record["loss_tokens"] != sum(mask):
        raise ValueError("declared supervision differs")
    return tokens, mask


def audit(source, output, max_length):
    output.mkdir(parents=True, exist_ok=False)
    count = tokens = supervision = 0
    lengths = []
    longest = None
    with source.open() as stream:
        for line in stream:
            record = json.loads(line)
            ids, mask = decode_record(record, max_length)
            if longest is None or len(ids) > len(longest["input_ids"]):
                longest = record
            count += 1
            tokens += len(ids)
            supervision += sum(mask)
            lengths.append(len(ids))
    if not count:
        raise ValueError("empty corpus")
    (output / "longest.jsonl").write_text(json.dumps(longest) + "\n")
    result = {"source": str(source.resolve()), "source_sha256": sha256(source),
              "documents": count, "input_tokens": tokens, "supervision": supervision,
              "minimum": min(lengths), "maximum": max(lengths),
              "estimated_cache_tensor_bytes": tokens * (6 * 4096 * 2 + 6)}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_cache(source, cache, max_length):
    """Check every document's exact token/mask boundary, without loading TB features."""
    import numpy as np
    from deepspec.data.target_cache_dataset import (
        CacheDataset,
        expected_target_cache_tensor_nbytes,
    )

    dataset = CacheDataset(str(cache))
    count = tokens = 0
    with source.open() as stream:
        for index, line in enumerate(stream):
            ids, mask = decode_record(json.loads(line), max_length)
            record = dataset._read_record(index)
            if record["seq_len"] != len(ids):
                raise ValueError(f"cache length differs at document {index}")
            shard = dataset._get_shard_mmap(record["shard_id"])
            cached_ids = np.frombuffer(shard, dtype=np.int32, count=len(ids),
                                       offset=record["input_ids_offset"]).tolist()
            cached_mask = np.frombuffer(shard, dtype=np.uint8, count=len(ids),
                                        offset=record["loss_mask_offset"]).tolist()
            if cached_ids != ids or cached_mask != mask:
                dataset.close()
                raise ValueError(f"cache token/mask differs at document {index}")
            del cached_ids, cached_mask
            sizes = expected_target_cache_tensor_nbytes(seq_len=len(ids), hidden_size=dataset.hidden_size,
                                                       num_target_layers=dataset.num_target_layers)
            for key, size in sizes.items():
                if record[f"{key}_offset"] + size > shard.size():
                    raise ValueError(f"truncated cache tensor {key} at document {index}")
            tokens += len(ids)
            count += 1
    if count != len(dataset):
        raise ValueError("cache document count differs")
    dataset.close()
    return {"documents": count, "input_tokens": tokens, "exact_token_and_loss_mask_match": True}


class PretokenizedCollator:
    """Reuse native CPU padding; fail instead of re-filtering or truncating."""

    def __init__(self, tokenizer, chat_template, max_length, min_loss_tokens):
        self.max_length = max_length

    def __call__(self, features):
        import torch
        from deepspec.data.target_cache_dataset import _pad_1d_batch

        records = []
        for feature in features:
            ids, mask = decode_record(feature, self.max_length)
            records.append({"input_ids": torch.tensor(ids, dtype=torch.long),
                            "attention_mask": torch.ones(len(ids), dtype=torch.long),
                            "loss_mask": torch.tensor(mask, dtype=torch.long)})
        return {key: _pad_1d_batch(records, key) for key in records[0]}


class TimedDataset:
    """Spawn-picklable observer delegating all decoding to native CacheDataset."""

    def __init__(self, cache_dir):
        from deepspec.data.target_cache_dataset import CacheDataset

        self.inner = CacheDataset(cache_dir)
        self.manifest = self.inner.manifest
        self.output = Path(os.environ["ASPEC_DEEPSPEC_READ_LOG"])
        self.rank = int(os.environ["ASPEC_DEEPSPEC_READ_RANK"])
        self.stream = None

    def __getstate__(self):
        return {**self.__dict__, "stream": None}

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        start = time.perf_counter()
        value = self.inner[index]
        elapsed = time.perf_counter() - start
        if self.stream is None:
            from multiprocessing.util import Finalize

            self.stream = (self.output / f"read-rank{self.rank}-pid{os.getpid()}.jsonl").open("a", buffering=1)
            Finalize(self, self.stream.close, exitpriority=10)
        self.stream.write(json.dumps([int(index), value["input_ids"].numel(), elapsed]) + "\n")
        return value


def check_origin(args):
    import deepspec
    if Path(deepspec.__file__).resolve().parent != args.checkout.resolve() / "deepspec":
        raise ValueError(f"unexpected DeepSpec origin: {deepspec.__file__}")


def prepare_worker(rank, args):
    import scripts.data.prepare_target_cache as native
    from transformers import AutoConfig, AutoModel

    check_origin(args)
    native.ConversationCollator = PretokenizedCollator
    original_factory = AutoModel.from_pretrained

    class TargetFactory:
        @staticmethod
        def from_pretrained(target, **kwargs):
            if args.max_length > 32768:
                config = AutoConfig.from_pretrained(target)
                # Same Target long-context policy as the frozen online study.
                config.rope_parameters = {"rope_type": "yarn", "factor": 4.0,
                                          "original_max_position_embeddings": 32768,
                                          "rope_theta": 1000000.0}
                config.max_position_embeddings = 131072
                kwargs["config"] = config
            return original_factory(target, **kwargs)

    native.AutoModel = TargetFactory
    sys.argv = ["prepare_target_cache", "--config", str(args.checkout / "config/dspark/dspark_qwen3_8b.py"),
                "--train-data-path", str(args.source), "--output-dir", str(args.output / "cache"),
                "--local-batch-size", str(args.local_batch), "--min-loss-tokens", "1",
                "--opts", f"model.target_model_name_or_path={json.dumps(str(args.target))}",
                "--opts", f"data.max_length={args.max_length}"]
    native.main(rank)


def train_worker(rank, args):
    import deepspec.data.cuda_prefetcher as native_prefetch
    import deepspec.trainer.base_trainer as native_base
    import deepspec.utils.training_logger as logger
    import torch
    import torch.distributed as dist
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
    from deepspec.utils import seed_all
    from deepspec.utils.config import (
        CustomJSONEncoder,
        load_config,
        parse_opts_to_config,
    )
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3Config

    check_origin(args)
    torch.set_float32_matmul_precision("high")
    seed_all(42)
    opts = [f"model.target_model_name_or_path={json.dumps(str(args.target))}",
            f"data.target_cache_path={json.dumps(str(args.cache))}", f"data.max_length={args.max_length}",
            "model.num_anchors=512", "train.num_train_epochs=1", "train.sharding_strategy=no_shard",
            f"train.global_batch_size={args.global_batch}", "train.local_batch_size=1"]
    if args.steps is not None:
        opts.append(f"train.max_train_steps={args.steps}")
    cfg = parse_opts_to_config(opts, load_config(args.checkout / "config/dspark/dspark_qwen3_8b.py"))
    cfg._origin_config_path = str(args.checkout / "config/dspark/dspark_qwen3_8b.py")
    cfg._origin_opts = opts
    cfg.logging.checkpoint_dir = str(args.output / "checkpoints")
    cfg.logging.tensorboard_dir = str(args.output / "tensorboard")
    geometry = json.loads(args.geometry.read_text())["draft_config"]
    observations = {"steps": [], "losses": [], "input_tokens": 0, "microbatches": 0,
                    "exposed_feed_wait_seconds": 0.0, "save_seconds": 0.0}
    loss_tensors = []
    transfer_events = []
    original_move = native_prefetch.move_batch_to_device

    def observe_transfer(batch, device):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_move(batch, device)
        end.record()
        transfer_events.append((begin, end))
        return result

    native_prefetch.move_batch_to_device = observe_transfer

    os.environ["ASPEC_DEEPSPEC_READ_LOG"] = str(args.output)
    os.environ["ASPEC_DEEPSPEC_READ_RANK"] = str(rank)
    native_base.CacheDataset = TimedDataset
    original_prefetch = native_base.CUDAPrefetcher

    class ObservedPrefetch(original_prefetch):
        def __iter__(self):
            start = time.perf_counter()
            result = super().__iter__()
            observations["exposed_feed_wait_seconds"] += time.perf_counter() - start
            return result

        def __next__(self):
            start = time.perf_counter()
            try:
                return super().__next__()
            finally:
                observations["exposed_feed_wait_seconds"] += time.perf_counter() - start

    native_base.CUDAPrefetcher = ObservedPrefetch
    original_log = logger.on_optimizer_step

    def log_step(**kwargs):
        result = original_log(**kwargs)
        observations["steps"].append({"step": kwargs["global_step"],
                                      "elapsed": time.perf_counter() - loop_start,
                                      "input_tokens": observations["input_tokens"],
                                      "microbatches": observations["microbatches"],
                                      "grad_norm": kwargs["grad_norm"]})
        return result

    logger.on_optimizer_step = log_step

    class OfflineTrainer(Qwen3DSparkTrainer):
        def build_models(self):
            config = Qwen3Config(**{**geometry, "num_anchors": 512, "block_size": 7,
                                   "num_target_layers": geometry["target_num_hidden_layers"]})
            config._attn_implementation = "flex_attention"
            model = Qwen3DSparkModel(config)
            state = adapt_state(load_file(str(args.initial), device="cpu"))
            index = args.target / "model.safetensors.index.json"
            shard = json.loads(index.read_text())["weight_map"]["lm_head.weight"]
            with safe_open(str(args.target / shard), framework="pt", device="cpu") as reader:
                state["lm_head.weight"] = reader.get_tensor("lm_head.weight")
            model.load_state_dict(state, strict=True)
            model.set_embedding_head_trainable(False)
            return model.to(device=self.device, dtype=self.precision_dtype), None

        def run_batch(self, batch):
            observations["input_tokens"] += batch["input_ids"].numel()
            observations["microbatches"] += 1
            loss = super().run_batch(batch)
            loss_tensors.append(loss.detach())
            return loss

        def save_and_eval_checkpoint(self):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = super().save_and_eval_checkpoint()
            torch.cuda.synchronize()
            observations["save_seconds"] += time.perf_counter() - start
            return result

    start = time.perf_counter()
    trainer = OfflineTrainer(rank, cfg)
    torch.cuda.synchronize()
    initialization = time.perf_counter() - start
    loop_start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_wall = time.perf_counter() - loop_start
    losses = torch.stack(loss_tensors).float().cpu().tolist()
    observations.update(losses=losses, initialization_seconds=initialization,
                        train_and_save_seconds=train_wall,
                        loop_seconds=train_wall - observations["save_seconds"],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        committed_steps=trainer.global_step,
                        h2d_stream_service_seconds=sum(a.elapsed_time(b) for a, b in transfer_events) / 1000,
                        finite=all(math.isfinite(x) for x in losses)
                        and all(math.isfinite(x["grad_norm"]) for x in observations["steps"]))
    if not observations["finite"]:
        raise ValueError("non-finite native training")
    (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(observations, indent=2) + "\n")
    if dist.get_rank() == 0:
        (args.output / "config.json").write_text(json.dumps(cfg, cls=CustomJSONEncoder, indent=2) + "\n")
    trainer.clean_up()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["audit", "prepare", "verify", "train"])
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--local-batch", type=int, default=16)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.phase in ("audit", "prepare", "verify") and (args.source is None or not args.source.is_file()):
        parser.error("--source must be a frozen JSONL file")
    if args.phase == "audit":
        print(json.dumps(audit(args.source, args.output, args.max_length), indent=2))
        return
    if args.phase == "verify":
        check_origin(args)
        if args.output.exists():
            raise FileExistsError(args.output)
        result = verify_cache(args.source, args.cache, args.max_length)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    import torch
    if not args.target or not args.target.is_dir():
        parser.error("--target must be a local model directory")
    if args.max_length not in (4096, 131072):
        parser.error("this Qwen3-8B study supports 4K and 128K")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        parser.error("preparation/training requires visible CUDA GPUs")
    if args.phase == "train":
        if any(path is None or not path.is_file() for path in (args.initial, args.geometry)):
            parser.error("--initial and --geometry must be files")
        if args.cache is None or not (args.cache / "manifest.json").is_file():
            parser.error("--cache must be a complete native cache")
        if args.global_batch < 1 or args.global_batch % gpu_count:
            parser.error("global batch must be positive and divisible by the Draft GPU count")
        if args.steps is not None and args.steps < 1:
            parser.error("--steps must be positive")
    elif args.local_batch < 1:
        parser.error("--local-batch must be positive")
    check_origin(args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"host": socket.gethostname(), "python": sys.executable, "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                "checkout_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "command": sys.argv, "started_unix": started, "adapter_sha256": sha256(Path(__file__))}
    for name in ("source", "initial", "geometry"):
        path = getattr(args, name)
        if path is not None:
            identity[f"{name}_sha256"] = sha256(path)
    (args.output / "attempt.json").write_text(json.dumps(identity, indent=2) + "\n")
    torch.multiprocessing.spawn(prepare_worker if args.phase == "prepare" else train_worker,
                                args=(args,), nprocs=torch.cuda.device_count())
    identity.update(finished_unix=time.time(), phase_seconds=time.time() - started)
    (args.output / "result.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
