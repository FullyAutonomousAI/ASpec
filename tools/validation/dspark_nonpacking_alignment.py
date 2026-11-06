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

"""Matched real-data DSpark training, using each framework's native optimizer.

Validation adapter only: one native ASpec producer supplies identical teacher
features to both trainers. The pinned TorchSpec sampler supplies replay anchors.
Neither framework's forward, objective, backward, normalization or update is
replaced. This is not an end-to-end performance comparison.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.validation.dspark_pipeline_alignment import file_sha, tensor_sha

REVISION = "a4e5c1085bfc5990ddcae6cf5832bb26380b38d3"
TRAIN_SHA = "ee84fefc24249c0f5e55f2bd82e8d61c12834b337d66e7bc017457f884e0e3c3"
TERMS = ("ce_loss", "l1_loss", "confidence_loss")


def prompt_identity(row):
    prompt = json.loads(row["source_metadata_json"])["prompt_input_ids"]
    if not prompt or row["input_ids"][: len(prompt)] != prompt:
        raise ValueError("Saved prompt is not the exact prefix of the saved training tokens")
    return hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()


def split_heldout(train_rows, heldout_rows, count=256):
    train_keys = {prompt_identity(row) for row in train_rows}
    ordered = sorted(
        ((prompt_identity(row), row) for row in heldout_rows), key=lambda item: item[0]
    )
    keys = [key for key, _ in ordered]
    if len(keys) != len(set(keys)) or train_keys.intersection(keys):
        raise ValueError("Duplicate or train/held-out overlapping prompt identity")
    if len(keys) <= count:
        raise ValueError("Acceptance must leave a nonempty independent validation split")
    return [row for _, row in ordered[count:]], [row for _, row in ordered[:count]]


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    root = Path(args.fixture)
    root.mkdir(parents=True, exist_ok=False)
    if file_sha(args.train) != TRAIN_SHA:
        raise ValueError("Training corpus differs from the frozen 60,416-document corpus")
    train = pq.read_table(args.train).to_pylist()
    heldout_table = pq.read_table(args.heldout)
    heldout = heldout_table.to_pylist()
    admitted = [row for row in heldout if row["loss_tokens"] >= 14]
    if len(train) != 60416 or sum(row["seq_len"] for row in train) != 44174395:
        raise ValueError("Frozen training count/tokens differ")
    if len({row["data_id"] for row in train}) != len(train):
        raise ValueError("Duplicate training identity")
    for row in train + admitted:
        if not 0 < row["seq_len"] <= 4096 or len(row["input_ids"]) != row["seq_len"]:
            raise ValueError("Invalid document length")
        from aspec.data.utils import unpack_loss_mask

        mask = unpack_loss_mask(row["packed_loss_mask"])
        prompt = json.loads(row["source_metadata_json"])["prompt_input_ids"]
        if (
            mask.numel() != row["seq_len"]
            or int(mask.sum()) != row["loss_tokens"]
            or mask[: len(prompt)].any()
        ):
            raise ValueError("Saved supervision mask/length/prompt boundary differs")
    validation, acceptance = split_heldout(train, admitted)
    validation_tail = validation[len(validation) // 4 * 4 :]
    validation = validation[: len(validation) // 4 * 4]
    pq.write_table(
        pa.Table.from_pylist(validation, schema=heldout_table.schema), root / "validation.parquet"
    )
    with (root / "acceptance.jsonl").open("w") as stream:
        for row in acceptance:
            stream.write(
                json.dumps(
                    {
                        "data_id": row["data_id"],
                        "prompt_input_ids": json.loads(row["source_metadata_json"])[
                            "prompt_input_ids"
                        ],
                    }
                )
                + "\n"
            )
    initial = Path(args.initial)
    config = OmegaConf.load(Path(__file__).parents[2] / "configs/recipes/qwen3_8b_dspark.yaml")
    config.model.target_model_path = args.target
    config.model.draft_model_config = str(initial / "config.json")
    config.model.initial_draft_model_path = str(initial)
    config.dataset.update(
        {
            "train_data_path": args.train,
            "eval_data_path": str(root / "validation.parquet"),
            "shuffle_dataset": False,
            "min_loss_tokens": 14,
            "eval_interval": 118,
        }
    )
    config.training.update(
        {
            "attention_backend": "cudnn_flex_attention",
            "dflash_packing": False,
            "lr_total_steps": 1180,
            "num_train_steps": 32,
            "num_epochs": 10,
            "min_lr": 0.0,
            "source_work_budget": None,
            "save_interval": 32,
            "save_final_checkpoint": True,
        }
    )
    config.output_dir = str(root / "run32")
    config.cache_dir = str(root / "cache")
    OmegaConf.save(config, root / "fast.yaml")
    manifest = {
        "torchspec_revision": REVISION,
        "train_rows": len(train),
        "train_tokens": 44174395,
        "train_sha256": TRAIN_SHA,
        "heldout_source_rows": len(heldout),
        "heldout_admitted_rows": len(admitted),
        "validation_rows": len(validation),
        "validation_unscored_dp_tail": [row["data_id"] for row in validation_tail],
        "acceptance_rows": len(acceptance),
        "split": "stable SHA256 of saved prompt token IDs; no regeneration",
        "prompt_overlap": 0,
        "lr_total_steps": 1180,
        "initial_files": {p.name: file_sha(p) for p in initial.iterdir() if p.is_file()},
        "draft_config": json.loads((initial / "config.json").read_text()),
        "teacher_boundary": "Shared actual producer tensors after frozen Target final norm",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def own_nonpacked(kwargs, width):
    """Own producer buffers; split the concatenation into zero-copy CPU views."""
    import torch

    if kwargs.get("ctx_doc_ids") is not None or kwargs["input_ids"].shape[0] != 1:
        raise ValueError("This protocol requires one non-packed document per microbatch")
    result = {}
    for key in (
        "input_ids",
        "loss_mask",
        "last_hidden_states",
        "injected_anchors",
        "injected_keep_mask",
    ):
        value = kwargs[key]
        if value is None:
            raise ValueError(f"Missing shared teacher/input field: {key}")
        result[key] = value.detach().cpu().clone()
    features = kwargs["hidden_states_list"]
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().clone()
        if features.shape[-1] % width:
            raise ValueError("Teacher capture width differs from the target hidden dimension")
        result["hidden_states_list"] = list(features.split(width, dim=-1))
    else:
        result["hidden_states_list"] = [value.detach().cpu().clone() for value in features]
    return result


def additive_terms(output):
    import torch

    numerator, denominator = output[6]
    return torch.stack(
        [
            numerator.detach().float(),
            denominator.detach().float(),
            *[output[5][key][0].detach().float() for key in TERMS],
        ]
    )


def normalized_terms(values):
    numerator, denominator, ce, l1, confidence = values
    if denominator <= 0 or not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite or empty optimizer window")
    result = dict(
        loss=numerator / denominator,
        denominator=denominator,
        ce=ce / denominator,
        l1=l1 / denominator,
        confidence=confidence / denominator,
    )
    if (
        abs(result["loss"] - (0.1 * result["ce"] + 0.9 * result["l1"] + result["confidence"]))
        > 1e-4
    ):
        raise ValueError("Additive components do not reconstruct the actual objective")
    return result


def position_totals(output):
    """Preserve additive token counts; never average document accuracies."""
    import torch

    count = output[4].detach().float()
    return torch.stack(
        (output[2].detach().float() * count, output[3].detach().float() * count, count)
    )


def training_acceptance(values, terms, acceptance=None):
    """Native teacher-forced proxies, not actual speculative decoding acceptance."""
    import torch

    ce_sum, correct, count = values.double()
    if (
        not torch.isfinite(values).all()
        or (count < 0).any()
        or (correct < -1e-4).any()
        or (correct > count + 1e-4).any()
        or count.sum() <= 0
    ):
        raise ValueError("Invalid pooled positional supervision")
    accuracy = correct / count.clamp(min=1)
    decay = torch.exp(-torch.arange(count.numel(), device=count.device).double() / 4)
    result = {
        "per_position_accuracy": accuracy.tolist(),
        "per_position_ce": (ce_sum / count.clamp(min=1)).tolist(),
        "per_position_count": count.tolist(),
        "token_accuracy": float(correct.sum() / count.sum()),
        "decay_weighted_accuracy": float((correct * decay).sum() / (count * decay).sum()),
        "simulated_accepted_draft_tokens": float(accuracy.cumprod(0).sum()),
        "weighted_distribution_overlap": 1 - 0.5 * terms["l1"],
        "convention": "legacy marginal-accuracy length excludes Target bonus; not decode acceptance",
    }
    if acceptance is not None:
        numerators, denominators = acceptance.double()
        if (
            acceptance.shape != (2, count.numel() + 1)
            or not torch.isfinite(acceptance).all()
            or (denominators < 0).any()
            or denominators[-1] <= 0
            or not torch.equal(denominators[:-1], count)
        ):
            raise ValueError("Invalid pooled probability acceptance supervision")
        rates = numerators / denominators.clamp(min=1)
        result.update(
            per_position_accept_rate=rates[:-1].tolist(),
            tau_probabilistic=float(rates[-1]),
            expected_accepted_draft_tokens=float(rates[-1] - 1),
            acceptance_block_count=float(denominators[-1]),
            acceptance_convention=(
                "alpha=1-L1/2; per-block prefix products then pooled; tau includes one Target token; "
                "fixed training trajectory, not real decode acceptance"
            ),
        )
    return result


def acceptance_totals(output):
    """Read small additive observations, without retaining vocabulary tensors."""
    import torch

    keys = [f"accept_rate@{index}" for index in range(output[4].numel())]
    keys.append("tau_probabilistic")
    return torch.stack([torch.stack(output[5][key]) for key in keys], dim=1).detach()


class ReferenceAcceptanceObserver:
    """Observe the native confidence target without changing upstream math.

    Only the pinned reference module's F binding is proxied; torch.nn.functional
    itself and every frozen source file stay untouched. The native BCE result is
    returned unchanged. Its detached target is already the native 1-L1/2 tensor.
    """

    def __init__(self, functional):
        self.functional = functional
        self.mask = None
        self.statistics = None

    def __getattr__(self, name):
        return getattr(self.functional, name)

    def binary_cross_entropy_with_logits(self, prediction, target, *args, **kwargs):
        from aspec.models.ops.acceptance import acceptance_statistics

        result = self.functional.binary_cross_entropy_with_logits(prediction, target, *args, **kwargs)
        if self.mask is None or self.statistics is not None:
            raise ValueError("Reference confidence observation is outside its single native forward")
        self.statistics = acceptance_statistics(target, self.mask)
        return result


def resumed_optimizer_audit(optimizer, step):
    """Observe actual native masters, moments and scheduler before any update."""
    import torch

    counts = [int(value["step"]) for value in optimizer.state.values()]
    if len(counts) != len(optimizer.model_params) or set(counts) != {step}:
        raise ValueError("Native resume did not restore every Adam update counter")
    if optimizer.lr_scheduler.last_epoch != step:
        raise ValueError("Native resume did not restore the LR phase")
    if not all(
        torch.equal(model, master.to(model.dtype))
        for model, master in zip(optimizer.model_params, optimizer.fp32_params, strict=True)
    ):
        raise ValueError("Native resumed BF16 weights disagree with their own FP32 masters")
    return {
        "step": step,
        "adam_slots": len(counts),
        "adam_steps": sorted(set(counts)),
        "learning_rate": optimizer.get_learning_rate(),
        "scheduler": optimizer.lr_scheduler.state_dict(),
        "masters_sha256": [tensor_sha(value) for value in optimizer.fp32_params],
        "moments_sha256": [
            {key: tensor_sha(value[key]) for key in ("exp_avg", "exp_avg_sq")}
            for value in optimizer.state.values()
        ],
    }


def vector_comparison(actual, reference):
    """Full tensor comparison, not a sampled norm or a difference of norms."""

    diff = actual_sq = ref_sq = 0.0
    max_abs = 0.0
    for actual_value, reference_value in zip(actual, reference, strict=True):
        if actual_value is None or reference_value is None:
            if actual_value is not None or reference_value is not None:
                raise ValueError("Gradient presence differs")
            continue
        val, ref = actual_value.detach().float(), reference_value.detach().float()
        if val.shape != ref.shape:
            raise ValueError("Compared tensor shapes differ")
        delta = val - ref
        diff += float(delta.square().sum())
        actual_sq += float(val.square().sum())
        ref_sq += float(ref.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    if not all(math.isfinite(value) for value in (diff, actual_sq, ref_sq, max_abs)):
        raise ValueError("Nonfinite compared tensors")
    return {
        "actual_norm": math.sqrt(actual_sq),
        "reference_norm": math.sqrt(ref_sq),
        "difference_norm": math.sqrt(diff),
        "relative_l2": math.sqrt(diff / max(ref_sq, 1e-30)),
        "max_absolute": max_abs,
    }


def optimizer_contract(optimizer):
    group = optimizer.optimizer.param_groups[0]
    return {key: group[key] for key in ("betas", "eps", "weight_decay", "fused")}


def materialize_reference_optimizer(optimizer, optimizer_dir):
    """Allocate the saved Adam destinations before upstream's native DCP load.

    Pinned TorchSpec exposes an empty lazy Adam state as the load template. DCP
    only reads destination keys, so that would silently skip saved moments. Reuse
    ASpec's metadata-validated destination allocation, without an optimizer
    update or a change to the upstream forward/backward/optimizer implementation.
    """
    from aspec.training.checkpoint import OptimizerState

    template = OptimizerState(optimizer.model, optimizer, load_path=Path(optimizer_dir))
    state = template._optimizer_load_template()
    optimizer.load_state_dict(state)
    return len(state["state"])


def reference_resume_location(path):
    """Map an explicit frontier to pinned TorchSpec's parent + ckpt_step API."""
    root = Path(path)
    if root.name.startswith("iter_"):
        try:
            step = int(root.name.removeprefix("iter_"))
        except ValueError as exc:
            raise ValueError("Invalid explicit TorchSpec checkpoint frontier") from exc
        if step < 1 or not (root / "model/.metadata").is_file():
            raise ValueError("Explicit TorchSpec checkpoint model is missing")
        return str(root.parent), step
    return str(root), None


@contextlib.contextmanager
def preserve_cutlass_compilation():
    """Keep the host compiler hooks while importing the frozen reference.

    TorchSpec's import-time disk-cache override forces caching even for hashless
    custom kernels. That breaks ASpec's fused Q/K compilation in a shared
    process. Restore the existing hooks, which cache stable modules and preserve
    upstream no-cache behavior for hashless modules. Neither model is changed.
    """
    from cutlass.base_dsl.compiler import CompileCallable
    from cutlass.base_dsl.dsl import BaseDSL

    compile_and_cache = BaseDSL.compile_and_cache
    compile_callable = CompileCallable._compile
    try:
        yield
    finally:
        BaseDSL.compile_and_cache = compile_and_cache
        CompileCallable._compile = compile_callable


class PairedReference:
    def __init__(self, trainer, fixture, run_dir, reference_resume=None, probe=True):
        import torch
        import torch.distributed as dist

        with preserve_cutlass_compilation():
            from torchspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
            from torchspec.models.dspark import DSparkModel
            from torchspec.training.dflash_trainer import DFlashTrainer
            from torchspec.training.fsdp import apply_fsdp2
            from torchspec.training.optimizer import BF16Optimizer

        self.trainer, self.root = trainer, Path(run_dir)
        self.rank = dist.get_rank()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((Path(fixture) / "manifest.json").read_text())
        self.width = manifest["draft_config"]["hidden_size"]
        if (
            trainer.args.fsdp_strategy != "REPLICATE"
            or trainer.args.dflash_packing
            or trainer.args.draft_context_parallel_size != 1
        ):
            raise ValueError("Actual execution is not the agreed nonpacking/REPLICATE/CP1 plan")
        expected = dict(
            lr_total_steps=1180,
            learning_rate=0.0006,
            min_lr=0.0,
            draft_accumulation_steps=128,
            micro_batch_size=1,
            warmup_ratio=0.04,
            max_grad_norm=1.0,
            weight_decay=0.0,
            seed=42,
            attention_backend="cudnn_flex_attention",
        )
        for key, value in expected.items():
            if getattr(trainer.args, key) != value:
                raise ValueError(f"Unexpected effective training setting {key}")
        if dist.get_world_size(trainer.dp_group) != 4:
            raise ValueError("The frozen GBS512 protocol requires four DP replicas")
        # Isolate constructor RNG: adding a reference must not change training RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            draft = DSparkDraftModel(DSparkConfig(**manifest["draft_config"]))
            draft.freeze_embedding()
            model = DSparkModel(
                draft,
                block_size=7,
                num_anchors=512,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )
            model.load_state_dict(trainer.model.state_dict(), strict=True)
            model.to(device="cuda", dtype=torch.bfloat16)
            self.model = apply_fsdp2(
                model, mesh=trainer.dp_mesh, args=trainer.args, modules_to_shard=list(draft.layers)
            )
        self.model.train()
        self.optimizer = BF16Optimizer(
            draft,
            lr=0.0006,
            min_lr=0.0,
            weight_decay=0.0,
            max_grad_norm=1.0,
            total_steps=1180,
            warmup_ratio=0.04,
        )
        if optimizer_contract(self.optimizer) != optimizer_contract(trainer.optimizer):
            raise ValueError("Native optimizer hyperparameters differ")
        fast_names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
        own_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if fast_names != own_names:
            raise ValueError("Trainable parameter identity/order differs")
        if any(p.dtype != torch.bfloat16 for p in self.optimizer.model_params):
            raise ValueError("Reference trainable parameters are not native BF16")
        if not reference_resume and not all(
            torch.equal(actual, reference)
            for actual, reference in zip(
                trainer.optimizer.fp32_params, self.optimizer.fp32_params, strict=True
            )
        ):
            raise ValueError("Initial FP32 masters differ")
        self.native = DFlashTrainer
        native_module = importlib.import_module("torchspec.models.dspark")
        self.reference_metrics = ReferenceAcceptanceObserver(native_module.F)
        native_module.F = self.reference_metrics
        self.proxy = SimpleNamespace(optimizer=self.optimizer, dp_group=trainer.dp_group)
        self.checkpoint_actor = SimpleNamespace(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.optimizer.lr_scheduler,
            global_step=trainer.global_step,
            args=SimpleNamespace(
                checkpoint_dir=str(self.root / "torchspec-checkpoints"),
                load_path=reference_resume,
                ckpt_step=None,
                continual_training=False,
            ),
        )
        if reference_resume:
            from torchspec.training import checkpoint

            self.checkpoint_actor.args.load_path, self.checkpoint_actor.args.ckpt_step = (
                reference_resume_location(reference_resume)
            )
            model_dir = checkpoint.resolve_resume_model_dir(self.checkpoint_actor.args)
            if model_dir is None:
                raise ValueError("Reference checkpoint is missing")
            saved_slots = materialize_reference_optimizer(
                self.optimizer, model_dir.parent / "optimizer"
            )
            state = checkpoint.load(self.checkpoint_actor)
            if state is None or state["metadata"]["global_step"] != trainer.global_step:
                raise ValueError("Reference resume is missing or at a different optimizer step")
            if len(self.optimizer.state) != saved_slots:
                raise ValueError("Reference resume did not restore every saved Adam state")
            if self.optimizer.get_learning_rate() != trainer.optimizer.get_learning_rate():
                raise ValueError("Native resumed learning rates differ")
            if self.rank == 0:
                audit = {
                    "aspec": resumed_optimizer_audit(trainer.optimizer, trainer.global_step),
                    "torchspec": resumed_optimizer_audit(self.optimizer, trainer.global_step),
                    "reference_checkpoint": str(model_dir.parent),
                    "aspec_checkpoint": trainer.args.load_path,
                }
                (self.root / "resume-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
                print(
                    "PAIRED_DSPARK_RESUME "
                    + json.dumps(
                        {
                            "step": trainer.global_step,
                            "adam_slots": saved_slots,
                            "lr": self.optimizer.get_learning_rate(),
                        }
                    ),
                    flush=True,
                )
            # Sampling is stateless; do not replace ASpec's restored training RNG.
        elif trainer.global_step:
            raise ValueError(
                "Cannot reconstruct the independent reference from ASpec resume weights"
            )
        self.rows, self.fast_totals = [], []
        self.fast_positions = []
        self.fast_acceptance = []
        self.eval_totals, self.reference_eval_totals = [], []
        self.eval_positions, self.reference_eval_positions = [], []
        self.eval_acceptance, self.reference_eval_acceptance = [], []
        self.eval_row = None
        self.probe = probe
        self.pending = None
        self.backend_calls = dict(forward=0, backward=0)
        from aspec.models.ops import cudnn_flex_attention as cudnn_module

        native_attention = cudnn_module.cudnn_flex_attention

        def observed_attention(*values, **options):
            output = native_attention(*values, **options)
            self.backend_calls["forward"] += 1
            if output.requires_grad:

                def backward(gradient):
                    self.backend_calls["backward"] += 1
                    return gradient

                output.register_hook(backward)
            return output

        cudnn_module.cudnn_flex_attention = observed_attention
        sources = {}
        for suffix in (
            "models.dspark",
            "models.draft.dspark",
            "models.draft.dflash",
            "models.ops.anchors",
            "training.optimizer",
            "training.dflash_trainer",
            "training.fsdp",
        ):
            module = importlib.import_module("torchspec." + suffix)
            sources[suffix] = {"path": module.__file__, "sha256": file_sha(module.__file__)}
        record = {
            "revision": REVISION,
            "sources": sources,
            "optimizer": optimizer_contract(self.optimizer),
            "optimizer_execution": {
                "aspec": {
                    "fused_clip_scaling": bool(trainer.optimizer._fuse_clip_scaling),
                },
                "torchspec": {"fused_clip_scaling": False},
            },
            "training": expected,
            "dp": 4,
            "gbs": 512,
            "probe": probe,
            "reference_namespace": importlib.import_module("torchspec").__file__,
            "aspec_namespace": importlib.import_module("aspec").__file__,
            "harness_sha256": file_sha(__file__),
            "adaptations": [
                "shared producer inputs",
                "exact native anchor replay",
                "observation only",
                "preserve host CUTLASS cache policy across reference imports",
                "allocate saved lazy Adam destinations before native reference resume",
            ],
        }
        (self.root / f"runtime-rank{self.rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    def pre_forward(self, module, positional, kwargs):
        import torch
        from torchspec.models.ops.anchors import sample_anchor_positions

        from aspec.models.ops.anchors import anchor_sampling_seed

        ids, mask = kwargs["input_ids"], kwargs["loss_mask"]
        training = torch.is_grad_enabled()
        seed = (
            anchor_sampling_seed(42, self.trainer.global_step, len(self.rows), self.rank)
            if training
            else int(tensor_sha(ids)[:15], 16)
        )
        with torch.random.fork_rng(devices=[ids.device]):
            torch.cuda.manual_seed(seed)
            anchors, keep = sample_anchor_positions(ids.shape[1], mask, 512, 7, ids.device)
        kwargs["injected_anchors"], kwargs["injected_keep_mask"] = anchors, keep
        self.lm_head = kwargs["lm_head_weight"].detach()
        if not training:
            self.eval_row = own_nonpacked(kwargs, self.width)
            return positional, kwargs
        self.rows.append(own_nonpacked(kwargs, self.width))
        trace = {
            "step": self.trainer.global_step + 1,
            "microbatch": len(self.rows) - 1,
            "input": tensor_sha(ids),
            "mask": tensor_sha(mask),
            "anchors": tensor_sha(anchors),
            "keep": tensor_sha(keep),
            "active_anchors": int(keep.sum()),
            "padded_tokens": ids.shape[1],
        }
        if self.probe:
            trace.update(
                features=[tensor_sha(v) for v in self.rows[-1]["hidden_states_list"]],
                teacher_last=tensor_sha(kwargs["last_hidden_states"]),
            )
        with (self.root / f"inputs-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(trace) + "\n")
        return positional, kwargs

    def post_forward(self, module, positional, output):
        import torch

        if torch.is_grad_enabled():
            self.fast_totals.append(additive_terms(output))
            self.fast_positions.append(position_totals(output))
            self.fast_acceptance.append(acceptance_totals(output))
        else:
            self.eval_totals.append(additive_terms(output))
            self.eval_positions.append(position_totals(output))
            self.eval_acceptance.append(acceptance_totals(output))
            was_training = self.model.training
            self.model.eval()
            try:
                reference_output = self.reference_forward(self.eval_row, synchronize=False)
                self.reference_eval_totals.append(additive_terms(reference_output))
                self.reference_eval_positions.append(position_totals(reference_output))
                self.reference_eval_acceptance.append(acceptance_totals(reference_output))
            finally:
                self.model.train(was_training)
                self.eval_row = None

    def finish_eval(self):
        if len(self.eval_totals) != 190 or len(self.reference_eval_totals) != 190:
            raise ValueError("Validation did not score the same fixed 760 documents")
        record = {
            "step": self.trainer.global_step,
            "criterion": "minimum pooled CE",
            "documents": 760,
            "fixed_anchors": True,
            "aspec": self.pooled(self.eval_totals, self.eval_positions, self.eval_acceptance),
            "torchspec": self.pooled(
                self.reference_eval_totals, self.reference_eval_positions, self.reference_eval_acceptance
            ),
        }
        if record["aspec"]["denominator"] != record["torchspec"]["denominator"]:
            raise ValueError("Validation supervision differs")
        with (self.root / f"validation-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.rank == 0:
            print("PAIRED_DSPARK_VALIDATION " + json.dumps(record), flush=True)
        self.eval_totals.clear()
        self.reference_eval_totals.clear()
        self.eval_positions.clear()
        self.reference_eval_positions.clear()
        self.eval_acceptance.clear()
        self.reference_eval_acceptance.clear()

    def reference_forward(self, row, *, synchronize):
        values = {
            key: ([v.to("cuda") for v in value] if isinstance(value, list) else value.to("cuda"))
            for key, value in row.items()
        }
        anchors = values.pop("injected_anchors")
        keep = values.pop("injected_keep_mask")
        # Reconstruct the frozen native contiguous supervision mask from the
        # exact replay inputs; no prediction, anchor or label is changed.
        import torch

        sequence = values["input_ids"].shape[1]
        labels = anchors.unsqueeze(-1) + torch.arange(1, 8, device=anchors.device)
        indices = torch.where(keep.unsqueeze(-1), labels.clamp(max=sequence - 1), 0)
        supervised = torch.gather(
            values["loss_mask"].unsqueeze(1).expand(-1, anchors.shape[1], -1), 2, indices
        )
        mask = keep.unsqueeze(-1) & (labels < sequence) & (supervised > 0.5)
        self.reference_metrics.mask = mask.int().cumprod(-1).bool()
        self.reference_metrics.statistics = None
        self.model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
        self.model.set_requires_gradient_sync(synchronize)
        output = self.model(**values, lm_head_weight=self.lm_head)
        if self.reference_metrics.statistics is None:
            raise ValueError("Native reference did not expose its confidence acceptance target")
        from aspec.models.ops.acceptance import acceptance_components

        components = {**output[5], **acceptance_components(self.reference_metrics.statistics)}
        self.reference_metrics.mask = None
        self.reference_metrics.statistics = None
        return (*output[:5], components, output[6])

    def pooled(self, totals, positions=None, acceptance=None):
        import torch
        import torch.distributed as dist

        values = torch.stack(totals).sum(dim=0)
        dist.all_reduce(values, group=self.trainer.dp_group)
        terms = normalized_terms(values.tolist())
        if positions is not None:
            pooled_positions = torch.stack(positions).sum(dim=0)
            dist.all_reduce(pooled_positions, group=self.trainer.dp_group)
            pooled_acceptance = None
            if acceptance is not None:
                pooled_acceptance = torch.stack(acceptance).sum(dim=0)
                dist.all_reduce(pooled_acceptance, group=self.trainer.dp_group)
            terms["training_acceptance"] = training_acceptance(
                pooled_positions, terms, pooled_acceptance
            )
        return terms

    def reference_window(self):
        self.optimizer.zero_grad()
        totals, positions, acceptance, denominator = [], [], [], None
        for index, row in enumerate(self.rows):
            output = self.reference_forward(row, synchronize=index == len(self.rows) - 1)
            self.native._backward(self.proxy, output[6][0], len(self.rows))
            totals.append(additive_terms(output))
            positions.append(position_totals(output))
            acceptance.append(acceptance_totals(output))
            value = output[6][1].detach()
            denominator = value.clone() if denominator is None else denominator + value
        self.native._normalize_accumulated_gradients(self.proxy, denominator, len(self.rows))
        return self.pooled(totals, positions, acceptance)

    def before_update(self):
        import torch

        if len(self.rows) != 128 or len(self.fast_totals) != len(self.rows):
            raise ValueError("Actual optimizer window differs from matched GBS512")
        fast = self.pooled(self.fast_totals, self.fast_positions, self.fast_acceptance)
        common = None
        if self.probe:
            self.model.load_state_dict(self.trainer.model.state_dict(), strict=True)
            if not all(
                torch.equal(value, self.model.state_dict()[key])
                for key, value in self.trainer.model.state_dict().items()
            ):
                raise ValueError("Same-weight reference copy is not bitwise exact")
            common = self.reference_window()
            common["gradients"] = vector_comparison(
                [p.grad for p in self.trainer.optimizer.model_params],
                [p.grad for p in self.optimizer.model_params],
            )
            common["per_parameter_gradients"] = {
                name: vector_comparison([actual.grad], [reference.grad])
                for (name, actual), (_, reference) in zip(
                    self.trainer.model.named_parameters(),
                    self.model.named_parameters(),
                    strict=True,
                )
                if actual.requires_grad
            }
        with torch.no_grad():
            for p, master in zip(
                self.optimizer.model_params, self.optimizer.fp32_params, strict=True
            ):
                p.copy_(master)
        independent = self.reference_window()
        independent["gradients"] = vector_comparison(
            [p.grad for p in self.trainer.optimizer.model_params],
            [p.grad for p in self.optimizer.model_params],
        )
        lr = self.optimizer.get_learning_rate()
        if lr != self.trainer.optimizer.get_learning_rate():
            raise ValueError("Actual learning rate phases differ")
        previous = None
        if self.probe:
            previous = (
                [p.detach().clone() for p in self.trainer.optimizer.fp32_params],
                [p.detach().clone() for p in self.optimizer.fp32_params],
            )
        norm = float(self.optimizer.step())
        independent["reported_gradient_norm"] = norm
        self.pending = {
            "step": self.trainer.global_step + 1,
            "learning_rate_before": lr,
            "aspec": fast,
            "common_weight": common,
            "torchspec": independent,
        }
        self.previous = previous

    def after_update(self, norm):

        record = self.pending
        record["aspec"]["reported_gradient_norm"] = float(norm)
        record["weights"] = vector_comparison(
            self.trainer.optimizer.fp32_params, self.optimizer.fp32_params
        )
        record["effective_backends"] = [
            {
                "attention": layer.self_attn.attention_backend,
                "qk_prepare": layer.self_attn.qk_prepare_backend,
            }
            for layer in self.trainer.draft_model.layers
        ]
        if any(
            value != {"attention": "cudnn_flex_attention", "qk_prepare": "aspec_cutedsl"}
            for value in record["effective_backends"]
        ):
            raise ValueError("Requested current cuDNN/QK optimizations did not actually execute")
        if min(self.backend_calls.values()) <= 0:
            raise ValueError("Missing actual cuDNN forward/backward execution")
        record["backend_calls"] = dict(self.backend_calls)
        if self.previous is not None:
            actual_old, ref_old = self.previous
            # Stream tensors instead of materializing a second full update vector.
            record["updates"] = vector_comparison(
                (
                    new.detach() - old
                    for new, old in zip(self.trainer.optimizer.fp32_params, actual_old, strict=True)
                ),
                (
                    new.detach() - old
                    for new, old in zip(self.optimizer.fp32_params, ref_old, strict=True)
                ),
            )
        if not all(math.isfinite(value) for value in record["weights"].values()):
            raise ValueError("Nonfinite native optimizer update")
        with (self.root / f"steps-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.rank == 0:
            print("PAIRED_DSPARK_STEP " + json.dumps(record), flush=True)
        self.rows.clear()
        self.fast_totals.clear()
        self.fast_positions.clear()
        self.fast_acceptance.clear()
        self.previous = None
        self.pending = None
        step = record["step"]
        if step == 32 or step % 118 == 0 or step == 1180:
            from torchspec.training import checkpoint

            self.checkpoint_actor.global_step = step
            checkpoint.save(self.checkpoint_actor, step - 1)


def run(args):
    from aspec.compat.sglang_patch import installed_site, patch_state

    # A historical venv can have the right SGLang version but an older training
    # patch. Reject it before allocating GPUs or beginning expensive controls.
    if patch_state(installed_site()) != "patched":
        raise ValueError("Apply the current ASpec SGLang training patch before this comparison")

    import aspec.train_entry as entry

    manifest = json.loads((Path(args.fixture) / "manifest.json").read_text())
    config_path = Path(args.fixture) / "fast.yaml"
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    if file_sha(config.dataset.train_data_path) != TRAIN_SHA:
        raise ValueError("Training corpus changed after preparation")
    for name, digest in manifest["initial_files"].items():
        if file_sha(Path(config.model.initial_draft_model_path) / name) != digest:
            raise ValueError(f"Shared initialization changed: {name}")
    reference_root = Path(args.reference_root).resolve()
    import torchspec

    if not Path(torchspec.__file__).resolve().is_relative_to(reference_root):
        raise ValueError("TorchSpec was not imported from the task's pinned reference checkout")
    expected = json.loads((reference_root / "alignment-source-hashes.json").read_text())
    if expected["revision"] != REVISION:
        raise ValueError("Reference revision differs")
    for relative, digest in expected["files"].items():
        if file_sha(reference_root / relative) != digest:
            raise ValueError(f"Pinned reference changed: {relative}")
    if manifest["train_sha256"] != TRAIN_SHA:
        raise ValueError("Unrecognized training fixture")
    base = entry.TrainerActor

    class AlignmentActor(base):
        def init(self, *positional, **kwargs):
            result = super().init(*positional, **kwargs)
            reference = PairedReference(
                self._trainer,
                args.fixture,
                args.run_dir,
                args.reference_resume,
                probe=not args.continuation,
            )
            self._alignment_reference = reference
            self._trainer.model.register_forward_pre_hook(reference.pre_forward, with_kwargs=True)
            self._trainer.model.register_forward_hook(reference.post_forward)
            original = self._trainer.optimizer.step
            original_eval = self._trainer.eval_from_cache

            def evaluate():
                metrics = original_eval()
                reference.finish_eval()
                return metrics

            self._trainer.eval_from_cache = evaluate

            def step(*values, **options):
                reference.before_update()
                norm = original(*values, **options)
                reference.after_update(norm)
                return norm

            self._trainer.optimizer.step = step
            return result

    entry.TrainerActor = AlignmentActor
    overrides = [f"output_dir={args.run_dir}"]
    if args.continuation:
        if not args.fast_resume or not args.reference_resume:
            raise ValueError("Continuation requires both independent native checkpoints")
        overrides += [
            "training.num_train_steps=1180",
            "training.save_interval=118",
            f"training.load_path={args.fast_resume}",
        ]
    sys.argv = ["aspec", "--config", str(config_path), *overrides]
    import uuid

    import ray

    # A pre-existing Ray service may run a different checkout/venv. Own this
    # validation cluster without stopping or changing any other cluster.
    ray.init(
        address="local",
        include_dashboard=False,
        object_store_memory=1024**3,
        _temp_dir=f"/tmp/fs-align-{uuid.uuid4().hex[:8]}",
    )
    try:
        entry.main()
    finally:
        ray.shutdown()


def compare(args):
    root = Path(args.run_dir)
    ranks = []
    for rank in range(4):
        records = [
            json.loads(line) for line in (root / f"steps-rank{rank}.jsonl").read_text().splitlines()
        ]
        if [row["step"] for row in records] != list(range(1, args.steps + 1)):
            raise ValueError("Missing or repeated real optimizer updates")
        ranks.append(records)
    rows = []
    for index, actual in enumerate(ranks[0]):
        for peer in ranks[1:]:
            # Native DP reduction gives every replica the same pooled objective
            # and model state. Do not sum replicated norms four times.
            if (
                peer[index]["aspec"] != actual["aspec"]
                or peer[index]["weights"] != actual["weights"]
            ):
                raise ValueError(
                    "Replicas disagree on the pooled loss or independently updated weights"
                )
        for mode in ("common_weight", "torchspec"):
            reference = actual[mode]
            if reference is None:
                raise ValueError("32-step gate is missing same-weight computation evidence")
            for key in ("loss", "ce", "l1", "confidence", "denominator"):
                val, ref = actual["aspec"][key], reference[key]
                if not math.isfinite(val) or not math.isfinite(ref):
                    raise ValueError("Nonfinite comparison")
                if key == "denominator" and val != ref:
                    raise ValueError("Effective supervision denominator differs")
                rows.append(
                    {
                        "step": actual["step"],
                        "mode": mode,
                        "metric": key,
                        "aspec": val,
                        "torchspec": ref,
                        "absolute_diff": abs(val - ref),
                        "relative_diff": abs(val - ref) / max(abs(ref), 1e-12),
                    }
                )
    summary = {
        "steps": args.steps,
        "documents": args.steps * 512,
        "status": "measurements complete; review numerical differences before full training",
        "metrics": {},
    }
    for mode in ("common_weight", "torchspec"):
        summary["metrics"][mode] = {
            key: {
                "max_absolute_diff": max(
                    row["absolute_diff"]
                    for row in rows
                    if row["mode"] == mode and row["metric"] == key
                ),
                "max_relative_diff": max(
                    row["relative_diff"]
                    for row in rows
                    if row["mode"] == mode and row["metric"] == key
                ),
            }
            for key in ("loss", "ce", "l1", "confidence", "denominator")
        }
        summary["metrics"][mode]["gradient_max_relative_l2"] = max(
            row[mode]["gradients"]["relative_l2"] for row in ranks[0]
        )
    for key in ("weights", "updates"):
        summary[key + "_max_relative_l2"] = max(row[key]["relative_l2"] for row in ranks[0])
    (root / "comparison.json").write_text(
        json.dumps({"summary": summary, "steps": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("fixture", "train", "heldout", "initial", "target"):
        prep.add_argument("--" + name, required=True)
    execution = sub.add_parser("run")
    for name in ("fixture", "run-dir", "reference-root"):
        execution.add_argument("--" + name, required=True)
    execution.add_argument("--continuation", action="store_true")
    execution.add_argument("--fast-resume")
    execution.add_argument("--reference-resume")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--run-dir", required=True)
    comparison.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
