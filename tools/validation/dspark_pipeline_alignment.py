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

"""Real Qwen3 target -> transport -> packed FSDP/AdamW loss replay.

This validation-only entry wraps the normal framework entry and TrainerActor.
Initial weights, anchor replay and observation are injected; production
data fetching, forward/backward, gradient normalization and optimizer are used.
An optional common-weight replay isolates same-state computation differences;
it does not establish independent optimizer-trajectory equivalence.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    import torch

    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def gradient_sample_indices(numel, device):
    import torch

    count = min(512, numel)
    if count <= 0:
        raise ValueError("Cannot sample an empty gradient")
    return torch.arange(count, device=device, dtype=torch.long) * (numel - 1) // max(count - 1, 1)


def replay_anchors(mask, docs, count, block, seed):
    import torch

    mask, docs = mask.cpu(), docs.cpu()
    anchors = torch.zeros((mask.shape[0], count), dtype=torch.long)
    keep = torch.zeros_like(anchors, dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)
    for row in range(mask.shape[0]):
        stop = max(mask.shape[1] - block, 0)
        valid = (
            (mask[row, :stop] > 0.5)
            & (mask[row, 1 : stop + 1] > 0.5)
            & (docs[row, :stop] >= 0)
            & (docs[row, :stop] == docs[row, block : block + stop])
        )
        candidates = valid.nonzero().flatten()
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
        chosen = chosen.sort().values
        anchors[row, : chosen.numel()] = chosen
        keep[row, : chosen.numel()] = True
    return anchors, keep


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel

    fixture = Path(args.fixture)
    fixture.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.draft_config).read_text())
    if config["hidden_size"] != 4096 or config["vocab_size"] != 151936:
        raise ValueError("This protocol requires the full Qwen3-8B draft dimensions")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    paragraphs = [
        "A distributed training system separates model semantics from execution. "
        "Each sample keeps its identity through packing and context parallelism. ",
        "The water cycle includes evaporation, condensation and precipitation. "
        "Energy conservation helps explain how temperature changes over time. ",
        "A balanced producer and consumer pipeline needs adequate buffers. "
        "A loss comparison must use the same teacher features and initial weights. ",
        "Mathematical reasoning benefits from checking assumptions and examples. "
        "For a matrix product, dimensions and accumulation precision both matter. ",
    ]
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_path = fixture / "data.parquet"
    rows = []
    lengths = (601, 911, 733, 797)
    for index in range(args.steps * 4 + 8):
        length = lengths[index % 4]
        text = f"Document {index}. " + paragraphs[index % 4] * 120
        ids = tokenizer.encode(text, add_special_tokens=False)[:length]
        if len(ids) != length:
            raise ValueError("Not enough tokens")
        mask = [0] * (length // (3 if index % 2 else 2)) + [1] * (
            length - length // (3 if index % 2 else 2)
        )
        mask[length * 3 // 4 : length * 3 // 4 + 3] = [0] * 3
        rows.append({"id": f"sample_{index}", "input_ids": ids, "loss_mask": mask})
    pq.write_table(pa.Table.from_pylist(rows), data_path)
    torch.manual_seed(731)
    model = DSparkDraftModel(DSparkConfig(**config)).to(torch.bfloat16)
    model.load_embedding(args.target, embedding_key="model.embed_tokens.weight")
    model.freeze_embedding()
    wrapper = DSparkModel(model, num_anchors=args.anchors, block_size=7)
    initial = fixture / "initial-state.pt"
    torch.save(wrapper.state_dict(), initial)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del wrapper, model
    for role in ("angel", "fast"):
        role_config = dict(config)
        role_config["model_type"] = "dspark" if role == "angel" else "qwen3_dspark"
        role_config["architectures"] = [
            "DSparkDraftModel" if role == "angel" else "Qwen3DSparkModel"
        ]
        (fixture / f"draft-{role}.json").write_text(json.dumps(role_config, indent=2) + "\n")
    common = {
        "model": {
            "target_model_path": args.target,
            "draft_model_config": str(fixture / "draft-fast.json"),
        },
        "dataset": {
            "train_data_path": str(data_path),
            "eval_data_path": None,
            "chat_template": None,
            "shuffle_dataset": False,
            "min_loss_tokens": 14,
        },
        "training": {
            "attention_backend": "fa4",
            "draft_context_parallel_size": 1,
            "dflash_packing": True,
            "micro_batch_size": 1,
            "draft_accumulation_steps": 2,
            "learning_rate": 0.0006,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "warmup_ratio": 0.04,
            "min_lr": 0.00006,
            "lr_total_steps": 100,
            "max_seq_length": 1536,
            "num_epochs": 1,
            "num_train_steps": args.steps,
            "seed": 731,
            "training_num_gpus_per_node": 1,
            "training_num_nodes": 1,
            "streaming_dispatch_with_training": True,
            "fsdp_strategy": "FULL_SHARD",
            "fsdp_reduce_dtype": "float32",
            "prefetch_depth": 1,
            "max_concurrent_batches": 1,
            "save_interval": 0,
            "save_per_epoch": False,
            "save_final_checkpoint": False,
            "dflash_block_size": 7,
            "dspark_num_anchors": args.anchors,
            "dspark_num_target_layers": config["num_target_layers"],
            "dspark_loss_decay_gamma": 4.0,
            "dspark_ce_loss_alpha": 0.1,
            "dspark_l1_loss_alpha": 0.9,
            "dspark_confidence_head_alpha": 1.0,
            "placement_strategy": "training_first",
        },
        "inference": {
            "inference_engine_type": "sgl",
            "store_last_hidden_states": True,
            "target_prefill_memory_optimized": False,
            "inference_num_gpus": 1,
            "inference_num_gpus_per_engine": 1,
            "inference_num_gpus_per_node": 8,
            "inference_batch_size": 2,
            "inference_fetch_batch": 2,
            "inference_buffer_threshold": 2,
            "max_sample_pool_size": 2,
            "sglang": {"tp_size": 1, "mem_fraction_static": 0.52},
        },
        "mooncake": {
            "data_plane": "transfer_engine",
            "protocol": "nvlink_intra",
            "require_nvlink": True,
            "enable_gpu_direct": True,
            "allow_host_fallback": False,
            "enable_hard_pin": False,
            "gpu_ring_slots": 18,
            "source_gpu_ring_slots": 18,
            "destination_gpu_ring_slots": 2,
            "max_seq_len": 1536,
            "hidden_dim": 4096,
            "global_segment_size": "4GB",
            "local_buffer_size": "1GB",
        },
        "logging": {"report_to": "none"},
        "debug": {"enable_perf_metrics": True},
        "output_dir": "",
        "cache_dir": str(fixture / "cache"),
    }
    common["model"]["native_context_length"] = 32768
    common["training"]["packing_min_samples_per_dispatch"] = 2
    OmegaConf.save(OmegaConf.create(common), fixture / "fast.yaml")
    common["mooncake"].update(
        data_plane="store",
        protocol="tcp",
        require_nvlink=False,
        enable_gpu_direct=False,
        allow_host_fallback=True,
    )
    common["model"].pop("native_context_length")
    common["dataset"]["chat_template"] = ""
    common["dataset"]["dataset_format"] = "pretokenized"
    common["training"].pop("packing_min_samples_per_dispatch")
    common["training"]["packing_min_fill_ratio"] = 0.90
    common["training"]["attention_backend"] = "flex_attention"
    common["model"]["draft_model_config"] = str(fixture / "draft-angel.json")
    OmegaConf.save(OmegaConf.create(common), fixture / "angel.yaml")
    manifest = {
        "steps": args.steps,
        "anchors": args.anchors,
        "target": args.target,
        "draft_config": config,
        "trainable_parameters": trainable,
        "initial_state_sha256": file_sha(initial),
        "data_sha256": file_sha(data_path),
    }
    (fixture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def install_initial_state(role, fixture):
    import torch
    import torch.distributed as dist

    module_name = (
        "aspec.training.dflash_trainer"
        if role == "fast"
        else ("angelspec.training.dspark_trainer")
    )
    module = importlib.import_module(module_name)
    original = module.fsdp2_load_full_state_dict

    def load(model, full_state, *args, **kwargs):
        if dist.get_rank() == 0:
            shared = torch.load(
                Path(fixture) / "initial-state.pt", map_location="cpu", weights_only=True
            )
            if shared.keys() != full_state.keys():
                raise ValueError("Initial state keys differ from the production model")
            for key in shared:
                if shared[key].shape != full_state[key].shape:
                    raise ValueError(f"Initial state shape differs: {key}")
            full_state = shared
        return original(model, full_state, *args, **kwargs)

    module.fsdp2_load_full_state_dict = load


def install_observers(trainer, role, run_dir, capture_trajectory=None, replay_trajectory=None):
    import torch
    import torch.distributed as dist

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    totals = []
    micro_index = 0
    original_aggregate = trainer._aggregate_metrics

    def pre_forward(module, positional, kwargs):
        nonlocal micro_index
        ids, mask, docs = kwargs["input_ids"], kwargs["loss_mask"], kwargs.get("ctx_doc_ids")
        if ids.shape[0] != 1 or docs is None:
            raise ValueError("Alignment requires one genuinely packed row per microbatch")
        anchors, keep = replay_anchors(
            mask,
            docs,
            module.num_anchors,
            module.block_size,
            731 + trainer.global_step * 1009 + micro_index,
        )
        kwargs["injected_anchors"] = anchors.to(ids.device)
        kwargs["injected_keep_mask"] = keep.to(ids.device)
        positions = kwargs.get("packed_context_positions")
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        lengths = [
            int((docs[0] == doc).sum()) for doc in docs[0].unique(sorted=True).tolist() if doc >= 0
        ]
        sampled = []
        for tensor in [*kwargs["hidden_states_list"], kwargs["last_hidden_states"]]:
            values = tensor.new_zeros((ids.shape[1], tensor.shape[-1]))
            owners = positions
            if tensor.shape[1] == ids.shape[1]:
                owners = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
                if dist.get_rank() != 0:
                    owners = owners[:, :0]
                    tensor = tensor[:, :0]
            if owners.shape[1] != tensor.shape[1]:
                raise ValueError("Feature ownership does not match its actual shape")
            values.index_copy_(0, owners.flatten(), tensor[0])
            sampled.append(values)
        features = torch.cat(sampled, dim=-1)
        dist.all_reduce(features)
        real = docs[0] >= 0
        trace = {
            "step": trainer.global_step + 1,
            "microbatch": micro_index,
            "document_lengths": lengths,
            "input_sha256": tensor_sha(ids[:, real]),
            "mask_sha256": tensor_sha(mask[:, real]),
            "anchors_sha256": tensor_sha(anchors),
            "keep_sha256": tensor_sha(keep),
            "active_anchors": keep.sum().item(),
            "feature_sha256": tensor_sha(features[real]),
            "input_tokens": int(real.sum()),
            "padded_tokens": ids.shape[1],
            "local_feature_tokens": positions.shape[1],
        }
        if dist.get_rank() == 0:
            with (run_dir / "microbatches.jsonl").open("a") as stream:
                stream.write(json.dumps(trace) + "\n")
        micro_index += 1
        return positional, kwargs

    def post_forward(module, positional, output):
        if role == "fast":
            numerator, denominator = output[-1]
            components = output[5]
        else:
            numerator, denominator, components = output[-1]
        totals.append(
            torch.stack(
                [
                    numerator.detach().float(),
                    denominator.float(),
                    *[
                        components[key][0].float()
                        for key in ("ce_loss", "l1_loss", "confidence_loss")
                    ],
                ]
            )
        )

    def aggregate(all_step_metrics, step, **kwargs):
        nonlocal micro_index
        metrics = original_aggregate(all_step_metrics, step, **kwargs)
        combined = torch.stack(totals).sum(dim=0)
        dist.all_reduce(combined)
        numerator, denominator, ce_num, l1_num, confidence_num = combined.tolist()
        loss = numerator / denominator
        components = {
            "ce": ce_num / denominator,
            "l1": l1_num / denominator,
            "confidence": confidence_num / denominator,
        }
        if not math.isfinite(loss) or denominator <= 0:
            raise ValueError("Nonfinite or empty optimizer window")
        reconstructed = 0.1 * components["ce"] + 0.9 * components["l1"] + components["confidence"]
        if abs(loss - reconstructed) > 1e-4:
            raise ValueError("Observed objective differs from additive components")
        if dist.get_rank() == 0:
            record = {
                "step": trainer.global_step,
                "loss": loss,
                **components,
                "denominator": denominator,
                "gradient_norm": metrics["train/grad_norm"],
                "learning_rate": metrics["train/lr"],
                "microbatches": micro_index,
                "optimizer": type(trainer.optimizer).__name__,
                "inner_optimizer": type(trainer.optimizer.optimizer).__name__,
                "model": type(trainer.model).__name__,
            }
            with (run_dir / "loss-steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print("ALIGNMENT_STEP " + json.dumps(record), flush=True)
        totals.clear()
        micro_index = 0
        return metrics

    original_train_step = trainer._train_step

    def train_step(*positional, **kwargs):
        if kwargs["batch_idx"] == 0 and (capture_trajectory or replay_trajectory):
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            directory = Path(capture_trajectory or replay_trajectory)
            directory.mkdir(parents=True, exist_ok=True)
            filename = directory / f"step-{trainer.global_step + 1:04d}.pt"
            if capture_trajectory:
                if filename.exists():
                    raise FileExistsError(f"Refusing to overwrite trajectory state: {filename}")
                state = get_model_state_dict(
                    trainer.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                )
                if dist.get_rank() == 0:
                    temporary = filename.with_suffix(".tmp")
                    torch.save(state, temporary)
                    temporary.replace(filename)
                dist.barrier()
            else:
                import time

                deadline = time.monotonic() + 180
                while not filename.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Missing common-weight trajectory state: {filename}")
                    time.sleep(0.2)
                state = (
                    torch.load(filename, map_location="cpu", weights_only=True)
                    if dist.get_rank() == 0
                    else {}
                )
                namespace = "aspec" if role == "fast" else "angelspec"
                loader = importlib.import_module(
                    namespace + ".training.fsdp"
                ).fsdp2_load_full_state_dict
                loader(trainer.model, state, trainer.dp_mesh, cpu_offload=None)
                trainer.optimizer.sync_fp32_params_from_model()
            if dist.get_rank() == 0:
                with (run_dir / "weight-trajectory.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": trainer.global_step + 1,
                                "state_sha256": file_sha(filename),
                                "mode": "capture" if capture_trajectory else "replay",
                            }
                        )
                        + "\n"
                    )
        return original_train_step(*positional, **kwargs)

    trainer._train_step = train_step

    original_optimizer_step = trainer.optimizer.step

    def observed_optimizer_step(*args, **kwargs):
        if trainer.global_step == 0:
            from torch.distributed.tensor import DTensor

            records = {}
            for name, parameter in trainer.model.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                gradient = gradient.full_tensor() if isinstance(gradient, DTensor) else gradient
                flat = gradient.detach().float().flatten()
                indices = gradient_sample_indices(flat.numel(), flat.device)
                records[name] = {
                    "norm": torch.linalg.vector_norm(flat).item(),
                    "numel": flat.numel(),
                    "sample": flat[indices].cpu().tolist(),
                }
            witness = {
                "global_norm": math.sqrt(sum(r["norm"] ** 2 for r in records.values())),
                "parameters": records,
            }
        result = original_optimizer_step(*args, **kwargs)
        if trainer.global_step == 0:
            witness["reported_norm"] = result.item()
            witness["norm_placements"] = [str(p) for p in getattr(result, "placements", ())]
            if dist.get_rank() == 0:
                (run_dir / "first-gradient.json").write_text(json.dumps(witness, indent=2) + "\n")
        return result

    trainer.optimizer.step = observed_optimizer_step

    trainer.model.register_forward_pre_hook(pre_forward, with_kwargs=True)
    trainer.model.register_forward_hook(post_forward)
    trainer._aggregate_metrics = aggregate


def validate_inputs(namespace, fixture, steps, cp, run_dir, learning_rate=None):
    config_module = importlib.import_module(namespace + ".config.train_config")
    overrides = [f"output_dir={run_dir}", f"training.num_train_steps={steps}"]
    if learning_rate is not None:
        overrides += [
            f"training.learning_rate={learning_rate}",
            f"training.min_lr={learning_rate / 10}",
        ]
    if namespace == "aspec":
        overrides += [
            f"training.draft_context_parallel_size={cp}",
            f"training.training_num_gpus_per_node={cp}",
        ]
    config = config_module.load_config(
        str(Path(fixture) / ("fast.yaml" if namespace == "aspec" else "angel.yaml")),
        cli_args=overrides,
        save_snapshot=False,
    )
    if namespace == "aspec":
        from aspec.planning import compile_execution_plan

        runtime = compile_execution_plan(config).to_runtime_args()
    else:
        runtime = config_module.config_to_flat_args(config)
        entry = importlib.import_module(namespace + ".train_entry")
        draft = importlib.import_module(namespace).AutoDraftModelConfig.from_file(
            runtime.draft_model_config
        )
        entry._resolve_batch_size(runtime)
        entry._validate_target_prefill_memory_optimized(runtime)
        entry._validate_and_configure_dflash(runtime, draft)
        entry._validate_packing_model_compatibility(runtime, draft)
        entry._validate_and_configure_dspark(runtime, draft)
    dataset = importlib.import_module(namespace + ".data.dataset").load_conversation_dataset(
        runtime
    )
    if len(dataset) < steps * 4:
        raise ValueError("Insufficient real source documents")
    print(f"ALIGNMENT_PREFLIGHT {namespace}: {len(dataset)} documents, CP{cp}", flush=True)
    return file_sha(config.dataset.train_data_path)


def run(args):
    role = args.role
    namespace = "aspec" if role == "fast" else "angelspec"
    entry = importlib.import_module(namespace + ".train_entry")
    base = entry.TrainerActor
    fixture, run_dir = args.fixture, args.run_dir

    class AlignmentActor(base):
        def init(self, *positional, **kwargs):
            install_initial_state(role, fixture)
            result = super().init(*positional, **kwargs)
            install_observers(
                self._trainer, role, run_dir, args.capture_trajectory, args.replay_trajectory
            )
            return result

    entry.TrainerActor = AlignmentActor
    overrides = [f"output_dir={run_dir}", f"training.num_train_steps={args.steps}"]
    if args.learning_rate is not None:
        overrides += [
            f"training.learning_rate={args.learning_rate}",
            f"training.min_lr={args.learning_rate / 10}",
        ]
    if role == "fast":
        overrides += [
            f"training.draft_context_parallel_size={args.cp}",
            f"training.training_num_gpus_per_node={args.cp}",
        ]
    elif args.cp != 1:
        raise ValueError("The repaired AngelSpec reference uses CP1")
    sys.argv = [namespace, "--config", str(Path(fixture) / f"{role}.yaml"), *overrides]
    Path(run_dir).mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(fixture) / "manifest.json").read_text())
    if file_sha(Path(fixture) / "initial-state.pt") != manifest["initial_state_sha256"]:
        raise ValueError("Shared initial weights changed")
    if (
        validate_inputs(namespace, fixture, args.steps, args.cp, run_dir, args.learning_rate)
        != manifest["data_sha256"]
    ):
        raise ValueError("Shared training data changed")
    if manifest["steps"] != args.steps:
        raise ValueError("Step protocol differs from the prepared fixture")
    metadata = {
        "role": role,
        "cp": args.cp,
        **manifest,
        "harness_sha256": file_sha(__file__),
        "learning_rate_override": args.learning_rate,
        "weight_trajectory": {"capture": args.capture_trajectory, "replay": args.replay_trajectory},
        "source_files": {},
    }
    for name in (
        "training.dspark_trainer",
        "training.dflash_trainer",
        "models.dspark",
        "models.draft.dspark",
        "training.optimizer",
        "training.fsdp",
        "training.trainer",
        "train_entry",
    ):
        module = importlib.import_module(namespace + "." + name)
        metadata["source_files"][module.__file__] = file_sha(module.__file__)
    (Path(run_dir) / "alignment-protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if role == "fast":
        entry.main()
    else:
        entry.train_async_no_generation(entry.parse_config())


def validate_weight_trajectory(root, roles, steps):
    paths = {role: root / role / "weight-trajectory.jsonl" for role in roles}
    if not any(path.exists() for path in paths.values()):
        return "independent-training"
    states = {}
    for role, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{role} is missing common-weight trajectory evidence")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        mode = "capture" if role == "angel" else "replay"
        if [record["step"] for record in records] != list(range(1, steps + 1)):
            raise ValueError(f"{role} has an incomplete common-weight trajectory")
        if any(record["mode"] != mode for record in records):
            raise ValueError(f"{role} has an unexpected common-weight trajectory mode")
        states[role] = [record["state_sha256"] for record in records]
    for role in roles[1:]:
        if states[role] != states["angel"]:
            raise ValueError(f"{role} common-weight trajectory mismatch")
    return "common-weight-replay"


def compare(args):
    root = Path(args.run_dir)
    roles = ("angel", "fast-cp1", "fast-cp2", "fast-cp4")
    protocol = validate_weight_trajectory(root, roles, args.steps)
    results, traces = {}, {}
    for role in roles:
        results[role] = [
            json.loads(line) for line in (root / role / "loss-steps.jsonl").read_text().splitlines()
        ]
        traces[role] = [
            json.loads(line)
            for line in (root / role / "microbatches.jsonl").read_text().splitlines()
        ]
        if len(results[role]) != args.steps or [r["step"] for r in results[role]] != list(
            range(1, args.steps + 1)
        ):
            raise ValueError(f"{role} has not completed every optimizer step")
        if any(
            not math.isfinite(record[key])
            for record in results[role]
            for key in ("loss", "ce", "l1", "confidence", "denominator", "gradient_norm")
        ):
            raise ValueError(f"{role} has non-finite loss/gradient evidence")
        if len(traces[role]) != args.steps * 2:
            raise ValueError(f"{role} has a different microbatch window")
    keys = (
        "step",
        "microbatch",
        "document_lengths",
        "input_sha256",
        "mask_sha256",
        "anchors_sha256",
        "keep_sha256",
        "feature_sha256",
        "active_anchors",
        "input_tokens",
    )
    for role in roles[1:]:
        for reference, candidate in zip(traces["angel"], traces[role], strict=True):
            if any(reference[key] != candidate[key] for key in keys):
                changed = [key for key in keys if reference[key] != candidate[key]]
                raise ValueError(
                    f"{role} step {reference['step']} input/replay mismatch: {changed}"
                )
    rows, summary = [], {"protocol": protocol, "optimizer_steps": args.steps}
    for role in roles[1:]:
        diffs = []
        for reference, candidate in zip(results["angel"], results[role], strict=True):
            absolute = abs(candidate["loss"] - reference["loss"])
            relative = absolute / max(abs(reference["loss"]), 1e-12)
            row = {
                "role": role,
                "step": reference["step"],
                "angel_loss": reference["loss"],
                "fast_loss": candidate["loss"],
                "absolute_diff": absolute,
                "relative_diff": relative,
                "relative_diff_percent": relative * 100,
            }
            for key in ("ce", "l1", "confidence", "denominator", "gradient_norm"):
                row[key + "_absolute_diff"] = abs(candidate[key] - reference[key])
                row[key + "_relative_diff"] = abs(candidate[key] - reference[key]) / max(
                    abs(reference[key]), 1e-12
                )
            rows.append(row)
            diffs.append(row)
        summary[role] = {
            "max_absolute_diff": max(r["absolute_diff"] for r in diffs),
            "max_relative_diff": max(r["relative_diff"] for r in diffs),
            "mean_relative_diff": sum(r["relative_diff"] for r in diffs) / args.steps,
        }
    for role in ("fast-cp2", "fast-cp4"):
        pairs = list(zip(results["fast-cp1"], results[role], strict=True))
        summary[role + "_vs_cp1"] = {
            "max_absolute_diff": max(abs(a["loss"] - b["loss"]) for a, b in pairs),
            "max_relative_diff": max(abs(a["loss"] - b["loss"]) / abs(a["loss"]) for a, b in pairs),
        }
    with (root / "loss-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("prepare", "angel", "fast", "compare"), required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft-config", default="aspec/config/dspark_draft_config.json")
    parser.add_argument("--capture-trajectory")
    parser.add_argument("--replay-trajectory")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--cp", type=int, default=1)
    args = parser.parse_args()
    if args.capture_trajectory and args.replay_trajectory:
        parser.error("Capture and replay are separate validation protocols")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive")
    if args.steps < 16:
        parser.error("Full-chain alignment requires at least 16 optimizer steps")
    if args.cp < 1 or args.anchors % args.cp:
        parser.error("Anchor count must be divisible by positive CP degree")
    if args.role == "prepare":
        prepare(args)
    elif args.role == "compare":
        compare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
