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

"""Bounded Draft-only CP1 benchmark using native trainers and shared real features.

Not an end-to-end framework benchmark or a training-quality test. Target generation,
disk reads, initialization and compilation warmup are outside the measured window.
GPU-resident source documents are collated inside each measured native training step.
The native anchor sampler runs, then a shared recorded selection controls equal work.
No numerical-oracle hooks, modified kernels or performance setting scans are used.
"""

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_lengths(capacity, shape):
    if shape == "single":
        return [capacity]
    # Unequal complete documents; approximately 95% fill, not adversarial padding.
    cuts = [int(capacity * ratio) for ratio in (0.15, 0.24, 0.30, 0.26)]
    return [value - index * 3 for index, value in enumerate(cuts)]


def configure_target_context(config, length):
    if length > 32768:
        # Transformers 5 stores theta inside rope_parameters; preserve it.
        config.rope_parameters = {
            **config.rope_parameters,
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 32768,
        }
        config.max_position_embeddings = length


def prepare(args):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    root = args.fixture
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.draft_config.read_text())
    if (config["hidden_size"], config["num_hidden_layers"]) != (4096, 5):
        raise ValueError("This first protocol requires the full five-layer 8B draft")
    target_config = AutoConfig.from_pretrained(args.target)
    configure_target_context(target_config, args.length)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModel.from_pretrained(
        args.target,
        config=target_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    captures = {}

    def hook_for(index):
        def capture(module, positional, kwargs):
            hidden = positional[0] if positional else kwargs["hidden_states"]
            captures[index] = hidden.detach().cpu()

        return capture

    handles = [
        target.layers[index].register_forward_pre_hook(hook_for(index), with_kwargs=True)
        for index in config["target_layer_ids"]
    ]
    try:
        for shape in ("single", "packed"):
            case = root / f"{args.length}-{shape}"
            case.mkdir(exist_ok=False)
            lengths = document_lengths(args.length, shape)
            documents = []
            for index, length in enumerate(lengths):
                text = (
                    f"Document {index}. "
                    "A distributed training system preserves document identity and context. "
                    "It checks attention visibility, numerical correctness and resource ownership. "
                )
                unit = tokenizer.encode(text, add_special_tokens=False)
                ids = torch.tensor((unit * math.ceil(length / len(unit)))[:length])[None]
                captures.clear()
                begin = time.perf_counter()
                with torch.inference_mode():
                    last = target(input_ids=ids.cuda(), use_cache=False).last_hidden_state.cpu()
                hidden = torch.cat([captures[layer] for layer in config["target_layer_ids"]], -1)
                mask = torch.ones_like(ids, dtype=torch.float32)
                mask[:, 0] = 0
                item = {
                    "input_ids": ids,
                    "loss_mask": mask,
                    "hidden_states": hidden,
                    "last_hidden_states": last,
                }
                path = case / f"document-{index}.pt"
                torch.save(item, path)
                documents.append({"path": path.name, "length": length, "sha256": file_sha(path)})
                print(
                    json.dumps({"prepared": str(path), "seconds": time.perf_counter() - begin}),
                    flush=True,
                )
                del item, hidden, last
            used = sum(lengths)
            # Shared anchors include no boundary crossing and all seven supervised slots.
            candidates = torch.cat(
                [
                    torch.arange(1, length - 7) + sum(lengths[:index])
                    for index, length in enumerate(lengths)
                ]
            )
            if candidates.numel() < 512:
                raise ValueError("Insufficient valid anchors")
            generator = torch.Generator().manual_seed(731)
            anchors = torch.stack(
                [
                    candidates[torch.randperm(candidates.numel(), generator=generator)[:512]]
                    .sort()
                    .values
                    for _ in range(args.warmup + args.steps)
                ]
            )
            torch.save(anchors, case / "anchors.pt")
            manifest = {
                "schema": "aspec.draft-only-thd-benchmark/v1",
                "capacity": args.length,
                "shape": shape,
                "lengths": lengths,
                "used_tokens": used,
                "fill_ratio": used / args.length,
                "documents": documents,
                "anchors": 512,
                "block_size": 7,
                "supervised_tokens_per_step": 3584,
                "steps": args.steps,
                "warmup": args.warmup,
                "anchors_sha256": file_sha(case / "anchors.pt"),
                "initial_state": str(args.initial.resolve()),
                "initial_sha256": file_sha(args.initial),
                "draft_config": config,
                "target": args.target,
                "feature_generation": "HF AutoModel SDPA, selected layer inputs, normalized final hidden",
                "target_config": target_config.to_dict(),
                "teacher_prenorm": False,
                "harness_sha256": file_sha(__file__),
            }
            (case / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        for handle in handles:
            handle.remove()


def apply_native_environment(namespace, root):
    # Use the native factory before torch imports, without importing torch twice.
    spec = importlib.util.spec_from_file_location(
        "draft_performance_actor_env", root / namespace / "utils" / "env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "get_aspec_env_vars", None) or module.get_angelspec_env_vars
    defaults = factory()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if key not in os.environ:
            os.environ[key] = defaults[key]


def run(args):
    namespace = "aspec" if args.role == "fast" else "angelspec"
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    apply_native_environment(
        namespace, args.repo.resolve() if args.repo else Path(__file__).resolve().parents[2]
    )
    import torch
    import torch.distributed as dist

    manifest = json.loads((args.case / "manifest.json").read_text())
    if args.result.exists():
        raise FileExistsError(args.result)
    if file_sha(manifest["initial_state"]) != manifest["initial_sha256"]:
        raise ValueError("Initial weights changed")
    if file_sha(args.case / "anchors.pt") != manifest["anchors_sha256"]:
        raise ValueError("Anchor fixture changed")
    documents = []
    for record in manifest["documents"]:
        path = args.case / record["path"]
        if file_sha(path) != record["sha256"]:
            raise ValueError("Target feature fixture changed")
        documents.append(
            {
                key: tensor.cuda()
                for key, tensor in torch.load(path, mmap=True, weights_only=True).items()
            }
        )
        if args.role == "fast":
            documents[-1]["_aspec_work"] = {
                "work_id": f"fixture-document-{len(documents) - 1}",
                "execution_id": f"fixture-document-{len(documents) - 1}-replay",
            }
    anchors = torch.load(args.case / "anchors.pt", weights_only=True).cuda()
    config_module = importlib.import_module(namespace + ".config.train_config")
    defaults = {
        **dataclasses.asdict(config_module.TrainingConfig()),
        **dataclasses.asdict(config_module.DebugConfig()),
    }
    defaults.update(
        target_model_path=manifest["target"],
        chat_template=None,
        attention_backend="fa4" if args.role == "fast" else "flex_attention",
        draft_context_parallel_size=1,
        dflash_packing=True,
        micro_batch_size=1,
        draft_accumulation_steps=1,
        max_seq_length=manifest["capacity"],
        dspark_num_anchors=512,
        dspark_num_target_layers=5,
        dflash_block_size=7,
        learning_rate=0.0006,
        min_lr=0.00006,
        max_grad_norm=1.0,
        weight_decay=0.0,
        warmup_ratio=0.04,
        lr_total_steps=100,
        fsdp_strategy="FULL_SHARD",
        fsdp_reduce_dtype="bfloat16",
        last_hidden_states_prenorm=False,
        load_path=None,
        resume=None,
        seed=731,
        enable_perf_metrics=True,
        checkpoint_dir=None,
    )
    dist.init_process_group(
        "nccl",
        init_method="file://" + str(args.result.with_suffix(".rendezvous").resolve()),
        rank=0,
        world_size=1,
    )
    importlib.import_module(namespace + ".utils.distributed").init_gloo_group()
    trainer_module = importlib.import_module(namespace + ".training.dspark_trainer")
    init_module = (
        trainer_module
        if args.role == "angel"
        else importlib.import_module(namespace + ".training.dflash_trainer")
    )
    original_load = init_module.fsdp2_load_full_state_dict

    def load_shared(model, state, *positional, **kwargs):
        state = torch.load(manifest["initial_state"], mmap=True, weights_only=True)
        return original_load(model, state, *positional, **kwargs)

    init_module.fsdp2_load_full_state_dict = load_shared
    trainer = trainer_module.DSparkTrainer(argparse.Namespace(**defaults))
    draft_config = dict(manifest["draft_config"])
    draft_config["max_position_embeddings"] = max(40960, manifest["capacity"])
    trainer.init_model(draft_config, manifest["target"])
    original_sample = trainer.dflash._sample_anchor_positions
    sample_index = 0

    def paired_sample(*positional, **kwargs):
        nonlocal sample_index
        original_sample(*positional, **kwargs)  # Pay the native sampler's normal cost.
        selection = anchors[sample_index][None]
        sample_index += 1
        return selection, torch.ones_like(selection, dtype=torch.bool)

    trainer.dflash._sample_anchor_positions = paired_sample
    collator_class = importlib.import_module(namespace + ".data.utils").DFlashPackingCollator
    collator = (
        collator_class(manifest["capacity"], pad_to_max_length=False)
        if args.role == "fast"
        else collator_class(manifest["capacity"])
    )

    class ResidentDocumentFeed:
        def __iter__(self):
            return self

        def __next__(self):
            return collator(documents)

        def iter_step_batches(self, count):
            for _ in range(count):
                yield next(self)

    trainer.data_fetcher = ResidentDocumentFeed()
    supervision_counts = []
    original_train_step = trainer._train_step
    sparse_builds = []
    sparse_build_count = 0
    if args.role == "fast":
        sparse_module = importlib.import_module("aspec.models.ops.thd_sparse_metadata")
        original_build = sparse_module.build_thd_sparse_pair

        def observe_build(**kwargs):
            nonlocal sparse_build_count
            sparse_build_count += 1
            return original_build(**kwargs)

        sparse_module.build_thd_sparse_pair = observe_build

    def observe_train_step(*positional, **kwargs):
        metrics = original_train_step(*positional, **kwargs)
        supervision_counts.append(metrics["count_per_position"])
        return metrics

    trainer._train_step = observe_train_step
    torch.manual_seed(731)
    results = []
    window_start = None
    try:
        for step in range(manifest["warmup"] + manifest["steps"]):
            if step == manifest["warmup"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                window_start = time.perf_counter()
            begin = time.perf_counter()
            sparse_build_count = 0
            metrics = trainer.train_from_queue(step=step, num_batches=1)
            sparse_builds.append(sparse_build_count)
            elapsed = time.perf_counter() - begin
            if (
                not metrics
                or not math.isfinite(metrics["train/avg_loss"])
                or not math.isfinite(metrics["train/grad_norm"])
            ):
                raise ValueError("Missing or nonfinite native training metrics")
            count = (
                sum(metrics[f"train/count_{index}"] for index in range(7))
                if "train/count_0" in metrics
                else None
            )
            record = {
                "step": step + 1,
                "measured": step >= manifest["warmup"],
                "wall_seconds": elapsed,
                "metrics": metrics,
                "supervised_count_from_metrics": count,
            }
            results.append(record)
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "measured": record["measured"],
                        "wall": elapsed,
                        "loss": metrics["train/avg_loss"],
                    }
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - window_start
        observed_counts = torch.stack(supervision_counts).float().cpu().tolist()
        if any(count != [512.0] * 7 for count in observed_counts):
            raise ValueError("Actual supervised work differs from the paired fixture")
        if args.role == "fast" and any(count != 1 for count in sparse_builds):
            raise ValueError("Native FSDP layers did not reuse one sparse pair per forward")
        for record, count in zip(results, observed_counts, strict=True):
            record["supervised_count_from_metrics"] = sum(count)
        measured = [item for item in results if item["measured"]]
        observations = [
            getattr(layer.self_attn, "attention_observation", {})
            for layer in trainer.draft_model.layers
        ]
        if args.role == "fast" and not all(
            item.get("qkv_format") == "thd" and item.get("backward_observed")
            for item in observations
        ):
            raise ValueError("THD forward/backward did not execute on every layer")
        source_files = {}
        for name in (
            "training.dspark_trainer",
            "training.dflash_trainer",
            "training.trainer",
            "training.optimizer",
            "training.fsdp",
            "models.dspark",
            "models.dflash",
            "models.draft.dspark",
            "models.draft.dflash",
            "data.utils",
            "models.ops.flex_attention",
            "utils.env",
        ):
            module = importlib.import_module(namespace + "." + name)
            source_files[module.__file__] = file_sha(module.__file__)
        if args.role == "fast":
            for name in ("direct_cp", "thd_attention", "fa4_sparse", "thd_sparse_metadata"):
                module = importlib.import_module(namespace + ".models.ops." + name)
                source_files[module.__file__] = file_sha(module.__file__)
        payload = {
            "role": args.role,
            "case": str(args.case),
            "manifest": manifest,
            "repeat": args.repeat,
            "wall_seconds": wall,
            "measured_steps": len(measured),
            "mean_step_seconds": wall / len(measured),
            "median_step_seconds": statistics.median(item["wall_seconds"] for item in measured),
            "input_tokens_per_second": manifest["used_tokens"] * len(measured) / wall,
            "supervised_tokens_per_second": manifest["supervised_tokens_per_step"]
            * len(measured)
            / wall,
            "documents_per_second": len(documents) * len(measured) / wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "attention_observations": observations,
            "steps": results,
            "source_files": source_files,
            "runtime_args": defaults,
            "harness_sha256": file_sha(__file__),
            "environment": {
                "host": socket.gethostname(),
                "torch": torch.__version__,
                "flash_attn_4": importlib.metadata.version("flash-attn-4"),
                "transformers": importlib.metadata.version("transformers"),
                "python_executable": sys.executable,
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "gpu_metadata": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id", "--format=csv"]
                ).decode(),
                "threads": {
                    key: os.environ.get(key)
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            },
            "observed_supervision_counts": observed_counts,
            "observed_sparse_pair_builds_per_step": sparse_builds if args.role == "fast" else None,
            "scope": "GPU-resident real Target features -> native collator, FSDP2 training and AdamW; no Target/transport timing",
        }
        args.result.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in (
                        "role",
                        "mean_step_seconds",
                        "input_tokens_per_second",
                        "peak_allocated_bytes",
                    )
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--draft-config", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--length", type=int, choices=(4096, 131072))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--role", choices=("angel", "fast"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
