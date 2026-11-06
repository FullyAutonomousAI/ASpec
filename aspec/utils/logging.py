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

import datetime
import json
import logging
import os
import threading
from pathlib import Path

import torch.distributed as dist

from aspec.utils import wandb as wandb_utils

_LOG_FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s %(message)s"
_tb_writer = None


def _get_logger_level():
    level_str = os.getenv("ASPEC_LOG_LEVEL", "INFO").upper()
    log_level = logging.getLevelNamesMapping().get(level_str)
    if log_level is None:
        logging.warning("Invalid log level: %s, defaulting to WARNING", level_str)
        return logging.WARNING
    return log_level


def setup_logger(log_level=None, actor_name=None, ip_addr=None):
    logger_name = "ASpec" if actor_name is None else f"ASpec-{actor_name}"
    _logger = logging.getLogger(logger_name)
    _logger.handlers.clear()
    _logger.propagate = False
    if log_level is None:
        log_level = _get_logger_level()
    _logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if ip_addr is None:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    else:
        rank = os.environ.get("RANK", 0)
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s{ip_addr} RANK:{rank}] %(filename)s:%(lineno)d %(levelname)s %(message)s"
            )
        )
    handler.setLevel(log_level)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


def setup_file_logging(
    role: str,
    rank: int | str,
    group: int = 0,
    log_dir: str | None = None,
) -> None:
    """Add a FileHandler to the module-level logger for per-role/per-node/per-rank file logging."""
    if log_dir is None:
        log_dir = os.environ.get("ASPEC_LOG_DIR")
    if log_dir is None:
        return

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    try:
        from aspec.utils.misc import get_current_node_ip

        node_ip = get_current_node_ip()
    except Exception:
        node_ip = "unknown"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(log_dir, role, node_ip)
    filename = f"{role}_g{group}_rank{rank}_{timestamp}.log"
    filepath = os.path.join(dir_path, filename)

    try:
        os.makedirs(dir_path, exist_ok=True)
        fh = logging.FileHandler(filepath)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        fh.setLevel(logger.level)
        logger.addHandler(fh)
        logger.info(f"File logging enabled: {filepath}")
    except OSError:
        logger.warning(f"Could not set up file logging at {filepath} — NFS may not be available")


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def init_tracking(args, primary: bool = True, **kwargs):
    global _tb_writer
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        if getattr(args, "use_tensorboard", False) and getattr(args, "output_dir", None):
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = os.path.join(args.output_dir, "runs")
            os.makedirs(tb_log_dir, exist_ok=True)
            _tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard writer initialized at {tb_log_dir}")
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def get_tb_writer():
    """Return the module-level TensorBoard SummaryWriter, or None if not initialized."""
    return _tb_writer


def close_tb_writer():
    """Flush and close the TensorBoard writer."""
    global _tb_writer
    if _tb_writer is not None:
        _tb_writer.flush()
        _tb_writer.close()
        _tb_writer = None


class ExperimentTracking:
    """Per-Draft ownership over the retained TorchSpec tracking adapters.

    Adapted from torchspec/utils/{logging,wandb}.py at
    6c042a87140a84d13839e341ece2c5c3ada918bc (MIT, notice above).
    Shared drivers use explicit handles, not global writers; every attempt owns
    new logs and retains its restored metric coordinates. No per-step file I/O.
    """

    def __init__(self, args):
        self.writer = None
        self.run = None
        self._closed = False
        self._lock = threading.RLock()
        self.path = Path(args.output_dir) / "tracking.json"
        self.record = {
            "schema_version": "aspec.experiment-tracking/v1",
            "run_id": args.aspec_run_id,
            "experiment": args.aspec_experiment_name,
            "state": "initializing",
            "resume_source": getattr(args, "load_path", None),
            "attempt_policy": "new_runs_with_restored_metric_steps",
            "tensorboard": {"enabled": False},
            "wandb": {"enabled": False},
            "metric_calls": 0,
            "last_train_step": None,
            "last_eval_step": None,
        }
        try:
            if getattr(args, "use_tensorboard", False):
                from torch.utils.tensorboard import SummaryWriter

                directory = Path(args.output_dir) / "runs"
                self.writer = SummaryWriter(log_dir=str(directory))
                self.record["tensorboard"] = {"enabled": True, "log_dir": str(directory)}
            self.run = wandb_utils.init_wandb_experiment(args)
            if self.run is not None:
                self.record["wandb"] = {
                    "enabled": self.run.settings.mode != "disabled",
                    "mode": self.run.settings.mode,
                    "run_id": self.run.id,
                    "project": self.run.project,
                    "group": self.run.group,
                    "name": self.run.name,
                    "directory": self.run.dir,
                }
            self.record["state"] = "active"
            self._persist()
        except BaseException as exc:
            self.record["error"] = f"{type(exc).__name__}: {exc}"
            try:
                self.close(exit_code=1)
            except Exception:
                logger.exception("Failed to close partially initialized experiment tracking")
            raise

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def log(self, metrics, *, train_step=None, inference_step=None, eval_step=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot write metrics after experiment tracking closed")
            payload = dict(metrics)
            for prefix, step in (("train", train_step), ("inference", inference_step), ("eval", eval_step)):
                if step is not None:
                    payload.setdefault(f"{prefix}/step", step)
            try:
                if self.run is not None:
                    self.run.log(payload)
                if self.writer is not None:
                    for key, value in payload.items():
                        if isinstance(value, (int, float)):
                            step = eval_step if key.startswith("eval/") else (
                                inference_step if key.startswith("inference/") else train_step
                            )
                            if step is not None:
                                self.writer.add_scalar(key, value, step)
            except Exception as exc:
                self.record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            self.record["metric_calls"] += 1
            if train_step is not None:
                self.record["last_train_step"] = train_step
            if eval_step is not None:
                self.record["last_eval_step"] = eval_step

    def close(self, *, exit_code=0):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            actions = []
            if self.writer is not None:
                actions.extend((self.writer.flush, self.writer.close))
            if self.run is not None:
                actions.append(lambda: self.run.finish(exit_code=exit_code or int(bool(errors))))
            for action in actions:
                try:
                    action()
                except Exception as exc:
                    errors.append(exc)
            self.record.update(state="failed" if errors or self.record.get("error") else "closed",
                               exit_code=exit_code or int(bool(errors)))
            if errors:
                self.record["close_errors"] = [f"{type(exc).__name__}: {exc}" for exc in errors]
            self._persist()
            if errors:
                raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close(exit_code=int(exc_type is not None))
        except Exception:
            if exc_type is None:
                raise
            logger.exception("Tracking close failed; preserving the original training error")
        return False
