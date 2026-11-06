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

"""Validation-only stock FA4 deterministic dense backward repeatability probe.

SM90 stock b27 does not implement deterministic block-sparse backward. This
diagnosis retains the fixed document mask but uses dense backward traversal;
it does not claim production sparse-path parity or change product defaults.
"""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace


def install_deterministic_fa4():
    import flash_attn.cute.interface as fa

    original = fa.flash_attn_func

    def deterministic(*args, **kwargs):
        kwargs["block_sparse_tensors_bwd"] = None
        kwargs["deterministic"] = True
        return original(*args, **kwargs)

    fa.flash_attn_func = deterministic
    return original


def install_fp32_ce():
    """Cast CE inputs only; retain BF16 projection, attention and model state."""
    from aspec.models import dspark

    original = dspark.F

    def cross_entropy(logits, *args, **kwargs):
        return original.cross_entropy(logits.float(), *args, **kwargs)

    # Do not patch torch.nn.functional globally or affect other loss terms.
    proxy = SimpleNamespace(**vars(original))
    proxy.cross_entropy = cross_entropy
    dspark.F = proxy
    return original


def kernel_probe(output):
    import flash_attn.cute.interface as fa
    import torch

    from aspec.models.ops.direct_cp import DirectCPMetadata
    from aspec.models.ops.thd_attention import thd_fa4_attention

    torch.manual_seed(731)
    device = "cuda"
    documents = torch.tensor([[0] * 601 + [1] * 911], device=device)
    anchors = torch.arange(128, device=device)[None] * 1504 // 127
    metadata = DirectCPMetadata(
        anchors,
        torch.ones_like(anchors, dtype=torch.bool),
        torch.arange(1512, device=device)[None],
        7,
        0,
        1,
        None,
        context_doc_ids=documents,
        anchor_doc_ids=documents.gather(1, anchors),
        sequence_length=1512,
    )
    inputs = [
        torch.randn(shape, device=device, dtype=torch.bfloat16)
        for shape in [(1, 32, 896, 128), (1, 8, 2408, 128), (1, 8, 2408, 128)]
    ]
    upstream = torch.randn_like(inputs[0])
    result = {}
    original = fa.flash_attn_func
    for mode in ("default_sparse", "deterministic_dense_backward"):
        if mode == "deterministic_dense_backward":
            install_deterministic_fa4()
        trials = []
        for trial in range(4):
            q, k, v = [value.clone().requires_grad_() for value in inputs]
            answer, _ = thd_fa4_attention(q, k, v, metadata, 128**-0.5)
            answer.backward(upstream)
            trials.append([answer.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()])
            print(f"FA4_REPEAT mode={mode} trial={trial + 1}", flush=True)
        fields = {}
        for index, field in enumerate(("output", "dq", "dk", "dv")):
            reference = trials[0][index].float()
            fields[field] = {
                "max_different_coordinates": max(
                    torch.count_nonzero(t[index].float() - reference).item() for t in trials[1:]
                ),
                "max_relative_l2": max(
                    (
                        (t[index].float() - reference).norm() / reference.norm().clamp_min(1e-30)
                    ).item()
                    for t in trials[1:]
                ),
            }
        result[mode] = fields
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        fa.flash_attn_func = original
    result["device"] = torch.cuda.get_device_name()
    result["fa4_source"] = fa.__file__
    result["trials_per_mode"] = 4
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-output")
    parser.add_argument("--fixture")
    parser.add_argument("--run-dir")
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument(
        "--fp32-ce", action="store_true", help="Validation-only CE precision isolation"
    )
    parser.add_argument(
        "--native-backward", action="store_true",
        help="Isolate FP32 CE while retaining the production sparse backward",
    )
    args = parser.parse_args()
    if args.native_backward and (args.kernel_output or not args.fp32_ce):
        parser.error("--native-backward requires a full-chain --fp32-ce diagnosis")
    if args.kernel_output:
        if Path(args.kernel_output).exists():
            raise FileExistsError(args.kernel_output)
        kernel_probe(args.kernel_output)
        return
    if not args.fixture or not args.run_dir:
        parser.error("A full-chain diagnostic needs --fixture and --run-dir")
    from aspec import train_entry

    base = train_entry.TrainerActor

    class PrecisionActor(base):
        def init(self, *positional, **kwargs):
            result = super().init(*positional, **kwargs)
            if not args.native_backward:
                install_deterministic_fa4()
            if args.fp32_ce:
                install_fp32_ce()
            return result

    train_entry.TrainerActor = PrecisionActor
    pipeline = runpy.run_path(str(Path(__file__).with_name("dspark_pipeline_alignment.py")))
    pipeline["run"](
        SimpleNamespace(
            role="fast",
            fixture=args.fixture,
            run_dir=args.run_dir,
            steps=16,
            cp=args.cp,
            learning_rate=None,
            capture_trajectory=None,
            replay_trajectory=None,
        )
    )
    (Path(args.run_dir) / "precision-diagnostic.json").write_text(
        json.dumps(
            {
                "validation_only": True,
                "override": (
                    "FP32 CE only / production FA4 sparse backward"
                    if args.native_backward else
                    "stock FA4 deterministic=True / dense fixed-mask backward"
                ),
                "production_sparse_backward_retained": args.native_backward,
                "establishes_sparse_path_parity": False,
                "production_default_changed": False,
                "ce_precision_changed": args.fp32_ce,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
