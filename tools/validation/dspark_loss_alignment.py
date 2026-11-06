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

"""Opt-in GPU DSpark parity: repaired Angel forward vs THD and direct CP reference.

Run prepare and angel from their respective repo roots, then fast via torchrun.
The fixture/result paths are explicit; this is not a benchmark or a runtime mode.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def reference_module():
    path = Path(__file__).with_name("dspark_reference.py")
    spec = importlib.util.spec_from_file_location("dspark_parity_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(path):
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel

    torch.manual_seed(731)
    config = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=97,
        target_hidden_size=256,
        target_num_hidden_layers=12,
        num_target_layers=2,
        target_layer_ids=[1, 8],
        mask_token_id=96,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000.0,
        fc_norm=False,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    model = DSparkDraftModel(DSparkConfig(**config))
    model.freeze_embedding()
    documents = []
    for index, length in enumerate([9, 17, 13, 7]):
        documents.append(
            {
                "input_ids": torch.randint(0, 96, (1, length)),
                "loss_mask": torch.ones(1, length),
                "hidden_states": torch.randn(1, length, 512),
                "last_hidden_states": torch.randn(1, length, 256),
                "_aspec_work": {"work_id": f"doc{index}", "execution_id": f"doc{index}-e0"},
            }
        )
    documents[1]["loss_mask"][0, 7] = 0  # global 16: an internal supervision gap
    anchors = torch.tensor([[0, 2, 4, 10, 12, 14, 17, 19, 27, 30, 32, 40] + [0] * 12])
    keep = torch.tensor([[True] * 12 + [False] * 12])
    torch.save(
        {
            "config": config,
            "state": model.state_dict(),
            "documents": documents,
            "lm_head_weight": torch.randn(97, 256) * 0.04,
            "anchors": anchors,
            "keep": keep,
            "block_size": 4,
            "row_capacity": 64,
        },
        path,
    )


def batch_from_fixture(fixture, role, rank, size, device):
    if role == "angel":
        from angelspec.data.utils import DFlashPackingCollator

        collator = DFlashPackingCollator(fixture["row_capacity"])
    else:
        from aspec.data.utils import DFlashPackingCollator

        collator = DFlashPackingCollator(
            fixture["row_capacity"],
            pad_to_max_length=False,
            context_parallel_rank=rank,
            context_parallel_size=size,
        )
    docs = []
    for item in fixture["documents"]:
        copy = dict(item)
        length = item["input_ids"].shape[1]
        start, end = length * rank // size, length * (rank + 1) // size
        copy["hidden_states"] = item["hidden_states"][:, start:end]
        copy["last_hidden_states"] = item["last_hidden_states"][:, start:end]
        docs.append(copy)
    row = collator(docs)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in row.items()
    }
    batch["lm_head_weight"] = fixture["lm_head_weight"].to(device, torch.bfloat16)
    batch["hidden_states_list"] = list(batch.pop("hidden_states").to(torch.bfloat16).split(256, -1))
    batch["last_hidden_states"] = batch["last_hidden_states"].to(torch.bfloat16)
    return batch


def kwargs_for_model(batch):
    return {
        key: batch[key]
        for key in (
            "input_ids",
            "hidden_states_list",
            "loss_mask",
            "lm_head_weight",
            "last_hidden_states",
            "ctx_doc_ids",
            "base_position_ids",
        )
    }


def build_model(fixture, role, size):
    if role == "angel":
        from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
        from angelspec.models.dspark import DSparkModel
    else:
        from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
        from aspec.models.dspark import DSparkModel
    draft = DSparkDraftModel(DSparkConfig(**fixture["config"]))
    draft.load_state_dict(fixture["state"], strict=True)
    draft.freeze_embedding()
    draft = draft.to(device="cuda", dtype=torch.bfloat16)
    extra = (
        {} if role == "angel" else dict(attention_backend="fa4", draft_context_parallel_size=size)
    )
    return DSparkModel(
        draft,
        block_size=4,
        num_anchors=24,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
        **extra,
    )


def run(args):
    size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if size > 1:
        dist.init_process_group("nccl")
        from aspec.utils.distributed import init_direct_cp_groups

        init_direct_cp_groups(size)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    device = torch.device("cuda", rank)
    batch = batch_from_fixture(fixture, args.role, rank, size, device)
    anchors, keep = fixture["anchors"].to(device), fixture["keep"].to(device)
    model = build_model(fixture, args.role, size)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    oracle = reference_module()
    expected = None
    if args.role == "fast":
        expected = (
            torch.load(args.reference_result, map_location="cpu", weights_only=True)
            if args.reference_result
            else None
        )
        if expected is not None and (
            expected["fixture_sha256"]
            != hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest()
        ):
            raise RuntimeError("Reference result belongs to a different fixture")
        if expected is not None and len(expected["steps"]) != args.steps:
            raise RuntimeError("Reference result has a different optimizer-step count")
    captured = {}
    model.draft_model.register_forward_hook(
        lambda module, inputs, output: captured.update(hidden=output)
    )
    records = []
    for step in range(args.steps):
        optimizer.zero_grad()
        if args.role == "angel":
            out = model(
                **kwargs_for_model(batch),
                injected_anchors=anchors,
                injected_keep_mask=keep,
                return_loss_terms=True,
            )
            hidden = captured["hidden"]
            terms = oracle.official_dspark_terms(hidden, model.draft_model, batch, anchors, keep, 4)
            if args.official_loss_source:
                oracle.check_unmodified_official_loss(args.official_loss_source, terms)
            official_loss = oracle.official_loss(terms)
            loss = out[0]
            torch.testing.assert_close(loss, official_loss, atol=0.002, rtol=0.002)
            supervised_counts = out[4].detach()
            torch.testing.assert_close(supervised_counts, terms["eval_mask"].float().sum((0, 1)))
            components = {
                name: (float(pair[0]), float(pair[1]))
                for name, pair in out[6][2].items()
                if name in ("ce_loss", "l1_loss", "confidence_loss")
            }
            for name, (num, den) in components.items():
                ref_num, ref_den = terms[name]
                torch.testing.assert_close(torch.tensor(den, device=device), ref_den)
                torch.testing.assert_close(
                    torch.tensor(num / den, device=device),
                    ref_num / ref_den,
                    atol=0.002,
                    rtol=0.002,
                )
            official_abs_diff = abs(float(loss.detach()) - float(official_loss.detach()))
            official_rel_diff = official_abs_diff / abs(float(official_loss.detach()))
            loss.backward()
        else:
            out = model(
                **kwargs_for_model(batch),
                packed_context_positions=batch.get("packed_context_positions"),
                injected_anchors=anchors,
                injected_keep_mask=keep,
            )
            hidden = captured["hidden"]
            numerator, denominator = out[6]
            numerator.backward()
            denominator = denominator.detach().clone()
            if size > 1:
                dist.all_reduce(denominator)
            for parameter in model.draft_model.parameters():
                if parameter.grad is not None:
                    if size > 1:
                        dist.all_reduce(parameter.grad)
                    parameter.grad.div_(denominator.clamp(min=1e-6))
            components = {}
            for name, pair in out[5].items():
                total = torch.stack(pair)
                if size > 1:
                    dist.all_reduce(total)
                components[name] = tuple(total.cpu().tolist())
            loss = sum(
                alpha * components[name][0] / components[name][1]
                for name, alpha in (("ce_loss", 0.1), ("l1_loss", 0.9), ("confidence_loss", 1.0))
            )
            supervised_counts = out[4].detach().clone()
            if size > 1:
                dist.all_reduce(supervised_counts)
            relative_error = loss_absolute_diff = loss_relative_diff = None
            if expected is not None:
                record = expected["steps"][step]
                loss_absolute_diff = abs(loss - record["loss"])
                loss_relative_diff = loss_absolute_diff / max(abs(record["loss"]), 1e-12)
                torch.testing.assert_close(supervised_counts.cpu(), record["supervised_counts"])
                first, last = 24 * rank // size, 24 * (rank + 1) // size
                active = keep[0, first:last].repeat_interleave(4)
                reference_hidden = record["hidden"][:, first * 4 : last * 4].to(device)
                torch.testing.assert_close(
                    hidden[:, active].float(),
                    reference_hidden[:, active].float(),
                    atol=0.015,
                    rtol=0.01,
                )
                torch.testing.assert_close(
                    torch.tensor(loss), torch.tensor(record["loss"]), atol=0.015, rtol=0.003
                )
                for name, (num, den) in components.items():
                    ref_num, ref_den = record["components"][name]
                    torch.testing.assert_close(torch.tensor(den), torch.tensor(ref_den))
                    torch.testing.assert_close(
                        torch.tensor(num / den),
                        torch.tensor(ref_num / ref_den),
                        atol=0.015,
                        rtol=0.003,
                    )
                errors, norms = [], []
                for name, parameter in model.draft_model.named_parameters():
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.float().cpu()
                    ref = record["gradients"][name]
                    torch.testing.assert_close(
                        grad, ref, atol=0.003, rtol=0.03, msg=lambda msg: name + ": " + msg
                    )
                    errors.append((grad - ref).flatten())
                    norms.append(ref.flatten())
                relative_error = float(torch.cat(errors).norm() / torch.cat(norms).norm())
                if relative_error > 0.025:
                    raise AssertionError(f"Whole gradient relative L2 error {relative_error}")
            observations = [
                layer.self_attn.attention_observation for layer in model.draft_model.layers
            ]
            assert all(o["qkv_format"] == "thd" and o["backward_observed"] for o in observations)
        gradients = {
            name: p.grad.float().cpu()
            for name, p in model.draft_model.named_parameters()
            if p.grad is not None
        }
        records.append(
            {
                "loss": float(loss.detach()) if torch.is_tensor(loss) else loss,
                "components": components,
                "gradients": gradients,
                "hidden": hidden.detach().cpu(),
                "supervised_counts": supervised_counts.cpu(),
                **(
                    {
                        "official_loss": float(official_loss.detach()),
                        "official_loss_absolute_diff": official_abs_diff,
                        "official_loss_relative_diff": official_rel_diff,
                    }
                    if args.role == "angel"
                    else {
                        "gradient_relative_l2_error": relative_error,
                        "loss_absolute_diff": loss_absolute_diff,
                        "loss_relative_diff": loss_relative_diff,
                    }
                ),
            }
        )
        optimizer.step()
    payload = {
        "role": args.role,
        "cp_size": size,
        "reference_result": args.reference_result,
        "angel_fp32_lm_head": getattr(model, "fp32_lm_head", None)
        if args.role == "angel"
        else None,
        "steps": records,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "dtype": "bfloat16",
        "model_source": str(Path(inspect.getfile(type(model))).resolve()),
        "model_source_sha256": hashlib.sha256(
            Path(inspect.getfile(type(model))).read_bytes()
        ).hexdigest(),
    }
    if rank == 0:
        torch.save(payload, args.result)
        summary = {
            **payload,
            "steps": [
                {
                    k: v
                    for k, v in item.items()
                    if k not in {"gradients", "hidden", "supervised_counts"}
                }
                for item in records
            ],
        }
        Path(args.result).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
    if size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["prepare", "angel", "fast"], required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--result")
    parser.add_argument("--reference-result", "--angel-result", dest="reference_result")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--official-loss-source")
    args = parser.parse_args()
    if args.role == "prepare":
        prepare(args.fixture)
    else:
        if not args.result or args.steps < 11:
            parser.error("validation requires --result and at least 11 optimizer steps")
        if args.role == "angel" and int(os.environ.get("WORLD_SIZE", "1")) != 1:
            parser.error("Angel reference must run on one GPU")
        run(args)
