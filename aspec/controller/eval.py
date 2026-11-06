# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Eval-specific setup and runtime helpers for the controller loop."""

import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from aspec.training.checkpoint import _read_checkpoint_metadata, _write_checkpoint_metadata
from aspec.utils.logging import get_tb_writer, logger

EVAL_CACHE_IDLE_TIMEOUT = 300.0


def eval_cache_identity(args, dataset_identity, count):
    """Bind reusable tensors to processed data, value semantics and rank layout."""
    if not isinstance(dataset_identity, str) or not dataset_identity.startswith("sha256:"):
        raise ValueError("eval cache requires a processed dataset fingerprint")
    target = Path(args.target_model_path).expanduser()
    if not target.is_dir():
        from transformers.utils.hub import cached_file

        target = Path(cached_file(str(target), "config.json", local_files_only=True)).parent
    snapshot = {}
    for path in sorted(target.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model"}:
            stat = path.stat()
            snapshot[path.name] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.suffix == ".json" else [stat.st_size, stat.st_mtime_ns]
            )
    fields = (
        "target_model_path", "max_seq_length", "aux_hidden_states_layers", "aspec_feature_contract",
        "aspec_context_contract", "chat_template", "renderer", "prompt_key", "eval_prompt_key",
        "dynamic_loss_mask", "last_turn_loss_only", "skip_after_header", "min_loss_tokens",
        "dp_size", "world_size", "draft_context_parallel_size", "attention_backend",
        "dflash_packing", "ttt_length", "seed",
        "sglang_extra_args", "sglang_tp_size", "sglang_pp_size",
        "target_prefill_memory_optimized",
    )
    # OmegaConf containers are converted at this boundary, not embedded in torch caches.
    from omegaconf import OmegaConf

    identity = {key: getattr(args, key, None) for key in fields}
    identity = OmegaConf.to_container(OmegaConf.create(identity), resolve=True)
    from aspec.runtime_identity import SGLANG_RUNTIME_RELATIVE_PATHS, module_runtime_identity

    versions = {}
    for name in ("sglang", "torch", "transformers", "flash-attn-4", "flashinfer-python", "sglang-kernel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema": "aspec.eval-cache/v2", "dataset": dataset_identity,
            "samples": count, "target_snapshot": snapshot, "inputs": identity,
            "producer_code": module_runtime_identity("sglang", relative_paths=SGLANG_RUNTIME_RELATIVE_PATHS)["effective_tree_sha256"],
            "consumer_cache_code": {name: module_runtime_identity(name)["effective_tree_sha256"]
                                    for name in ("aspec.training.data_fetcher", "aspec.utils.processing",
                                                 "aspec.data.utils")},
            "versions": versions}


@dataclass
class EvalSetupState:
    eval_interval: int
    eval_enabled: bool
    eval_cache_loaded: bool
    eval_cache_path: str | None
    best_eval_score: float
    eval_dispatch_bs: int
    eval_dataset_size: int
    dp_size: int
    initial_eval_submit_count: int = 0
    cache_identity: dict | None = None
    history: list = field(default_factory=list)
    cache_idle_timeout_seconds: float | None = None


def _check_idle_timeout(
    dispatched_samples: int, last_progress_at: float, total_samples: int,
    timeout_seconds: float | None = None,
) -> None:
    timeout_seconds = EVAL_CACHE_IDLE_TIMEOUT if timeout_seconds is None else timeout_seconds
    idle_for = time.monotonic() - last_progress_at
    if idle_for >= timeout_seconds:
        raise TimeoutError(
            "Timed out while waiting for eval cache generation "
            f"(no progress during eval for {idle_for:.1f}s, "
            f"idle_timeout={timeout_seconds:.1f}s, "
            f"dispatched={dispatched_samples}/{total_samples} samples)"
        )


def eval_selection_metric(metrics):
    """Retrieval recall is not decoder acceptance; each stage selects its own proxy."""
    return "eval/candidate_top1_recall" if "eval/candidate_top1_recall" in metrics else "eval/simulated_acc_len"


def update_checkpoint_eval_meta(
    checkpoint_dir: str | None,
    step: int,
    eval_metrics: dict,
    current_best: float,
) -> float:
    """Append eval metrics to checkpoint meta.json and track best checkpoint."""
    if not checkpoint_dir or not eval_metrics:
        return current_best

    base_dir = Path(checkpoint_dir).expanduser()
    step_id = step + 1
    meta_path = base_dir / f"iter_{step_id:07d}" / "meta.json"

    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}, skipping eval meta update")
        return current_best

    for key in ("eval/avg_loss", "eval/avg_acc", "eval/simulated_acc_len",
                "eval/candidate_top1_recall", "eval/candidate_teacher_mass"):
        if key in eval_metrics:
            metadata[key] = eval_metrics[key]
    _write_checkpoint_metadata(meta_path, metadata)

    selection_metric = eval_selection_metric(eval_metrics)
    score = eval_metrics.get(selection_metric)
    if score is not None and math.isfinite(score) and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} ({selection_metric}={score:.2f})")

    return current_best


def generate_eval_cache(
    controller,
    inference_manager,
    train_group,
    eval_state: EvalSetupState,
    *,
    check_producer=None,
) -> None:
    """Drain eval results from inference and cache on each trainer."""
    eval_dataset_size = eval_state.eval_dataset_size
    dp_size = eval_state.dp_size
    dispatch_bs = eval_state.eval_dispatch_bs
    eval_cache_path = eval_state.eval_cache_path

    last_progress_at = time.monotonic()
    logger.info(
        f"Caching eval hidden states from inference engine "
        f"({eval_dataset_size} samples, dispatch_bs={dispatch_bs})..."
    )
    transition = ray.get(
        inference_manager.start_eval_budget.remote(eval_dataset_size)
    )
    if transition.get("phase") != "eval":
        raise RuntimeError(f"failed to start eval budget: {transition}")
    if eval_state.initial_eval_submit_count <= 0:
        raise RuntimeError("eval cache generation requires an initial submitted chunk")
    ray.get(
        controller.submit_eval_chunk.remote(
            0, eval_state.initial_eval_submit_count
        )
    )
    dispatched_samples = 0
    next_submit_idx = eval_state.initial_eval_submit_count
    eval_progress = tqdm(total=eval_dataset_size, desc="Eval caching", unit="sample")

    while dispatched_samples < eval_dataset_size:
        if check_producer is not None:
            check_producer()
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            if check_producer is None:
                train_group.cache_eval_samples(dispatch_bs // dp_size)
            else:
                train_group.cache_eval_samples(dispatch_bs // dp_size, check_producer=check_producer)
            dispatched_samples += dispatch_bs
            if next_submit_idx < eval_dataset_size:
                next_end = min(next_submit_idx + dispatch_bs, eval_dataset_size)
                ray.get(controller.submit_eval_chunk.remote(next_submit_idx, next_end))
                next_submit_idx = next_end
            last_progress_at = time.monotonic()
            eval_progress.update(dispatch_bs)
        else:
            _check_idle_timeout(dispatched_samples, last_progress_at, eval_dataset_size,
                                eval_state.cache_idle_timeout_seconds)
            time.sleep(0.01)

    eval_progress.close()

    ray.get(controller.finalize_eval_dispatch.remote())
    restored = ray.get(inference_manager.finish_eval_budget.remote())
    logger.info(
        "Eval budget drained; restored %s budget (%s samples)",
        restored.get("phase"),
        restored.get("sample_budget"),
    )
    logger.info(f"Eval caching complete ({dispatched_samples} samples)")
    if eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


def run_eval(step: int, train_group, eval_enabled: bool, *, state=None, args=None, tracking=None) -> dict:
    """Run forward-only eval from cache. Assumes eval cache is already populated."""
    if not eval_enabled:
        return {}
    eval_results = train_group.run_eval()
    if not eval_results or any(not result for result in eval_results):
        raise RuntimeError("Requested held-out eval produced no metrics on one or more ranks")
    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        if not all(math.isfinite(value) for value in eval_metrics.values() if isinstance(value, (int, float))):
            raise RuntimeError("Non-finite held-out evaluation metrics")
        eval_metrics["eval/step"] = step
        if tracking is not None:
            tracking.log(eval_metrics, eval_step=step)
        else:
            if wandb.run is not None:
                wandb.log(eval_metrics)
            writer = get_tb_writer()
            if writer is not None:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
                writer.flush()
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"{eval_selection_metric(eval_metrics)}={eval_metrics.get(eval_selection_metric(eval_metrics), 0):.2f}"
        )
        if state is not None and args is not None:
            checkpoint_dir = getattr(args, "checkpoint_dir", None)
            record = {"step": step, "metrics": dict(eval_metrics),
                      "checkpoint": str(Path(checkpoint_dir) / f"iter_{step + 1:07d}") if checkpoint_dir else None}
            state.history = [r for r in state.history if r["step"] != step] + [record]
            output = Path(args.output_dir) / "evaluation.json"
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "schema": "aspec.evaluation/v1", "cache_identity": state.cache_identity,
                "selection_metric": eval_selection_metric(eval_metrics), "proxy_only": True,
                "history": state.history,
            }, indent=2, allow_nan=False) + "\n")
            os.replace(temporary, output)
    return eval_metrics


def setup_eval(controller, train_group, args, eval_dataset_size: int) -> EvalSetupState:
    """Prepare eval runtime settings and optionally load/submit eval cache input."""
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    eval_cache_loaded = False
    initial_eval_submit_count = 0
    identity = None
    history = []

    eval_dispatch_bs = min(args.dp_size, eval_dataset_size)

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if eval_selection_metric(existing) in existing:
                best_eval_score = existing[eval_selection_metric(existing)]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    if eval_enabled:
        cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
        admission = ray.get(controller.get_dataset_admission_evidence.remote())
        identity = eval_cache_identity(args, admission["eval"]["dataset_fingerprint"], eval_dataset_size)
        if getattr(args, "load_path", None) and not getattr(args, "continual_training", False):
            from aspec.training.checkpoint import resolve_resume_model_dir

            source = resolve_resume_model_dir(args)
            previous = Path(getattr(args, "resume_source_run_dir", Path(source).parent.parent.parent)) / "evaluation.json" if source else None
            if previous and previous.is_file():
                retained = json.loads(previous.read_text())
                if retained.get("cache_identity") == identity:
                    frontier = _read_checkpoint_metadata(Path(source).parent / "meta.json")["global_step"]
                    history = [r for r in retained["history"] if r["step"] <= frontier]
                    eligible = [r["metrics"].get(eval_selection_metric(r["metrics"]), -math.inf)
                                for r in history if r.get("checkpoint") and Path(r["checkpoint"]).is_dir()]
                    best_eval_score = max([best_eval_score, *eligible])
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

        loaded = train_group.load_eval_cache(eval_cache_path, identity=identity)
        if loaded and all(n > 0 and n == eval_dataset_size // args.dp_size for n in loaded):
            eval_cache_loaded = True
            logger.info(
                f"Eval: loaded cached tensors from {eval_cache_path} ({loaded[0]} batches per rank)"
            )
        else:
            train_group.clear_eval_cache()
            inference_batch_size = args.inference_batch_size
            initial_eval_submit_count = min(
                eval_dataset_size,
                max(eval_dispatch_bs * 2, inference_batch_size),
            )
            logger.info(
                f"Eval: {eval_dataset_size} samples, dispatch_bs={eval_dispatch_bs}, "
                f"initial_submit={initial_eval_submit_count}"
            )

    return EvalSetupState(
        cache_idle_timeout_seconds=getattr(args, "eval_cache_idle_timeout_seconds", None),
        eval_interval=eval_interval,
        eval_enabled=eval_enabled,
        eval_cache_loaded=eval_cache_loaded,
        eval_cache_path=eval_cache_path,
        best_eval_score=best_eval_score,
        eval_dispatch_bs=eval_dispatch_bs,
        eval_dataset_size=eval_dataset_size,
        dp_size=args.dp_size,
        initial_eval_submit_count=initial_eval_submit_count,
        cache_identity=identity,
        history=history,
    )
