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

"""GPU forward/backward comparison against isolated official Ling FFN classes."""
import argparse
import json
import runpy
from pathlib import Path

import torch

from aspec.models.draft.dflash import DFlashConfig
from aspec.models.draft.ling_moe import LingDraftMoE
from tools.validation.moe_draft_distributed import error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = runpy.run_path(str(Path(__file__).parents[2] / "tests/reference_ling_moe.py"))
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for top_k in (1, 2, 4):
            torch.manual_seed(78)
            cfg = DFlashConfig(hidden_size=32, intermediate_size=64, hidden_act="silu",
                               num_hidden_layers=1, ffn_type="ling_moe", num_experts=8,
                               num_experts_per_tok=top_k, moe_intermediate_size=24,
                               num_shared_experts=2, moe_shared_expert_intermediate_size=12,
                               n_group=2, topk_group=1)
            model = LingDraftMoE(cfg).cuda().to(dtype)
            with torch.no_grad():
                model.gate.expert_bias.copy_(torch.linspace(-.2, .2, 8, device="cuda"))
            reference = official["BailingMoeV3SparseMoeBlock"](cfg).cuda().to(dtype)
            # Public HF casts its expert-bias buffer along with parameters.
            # Restore FP32 state before copying, as in the training reference.
            reference.gate.expert_bias = reference.gate.expert_bias.float()
            reference.load_state_dict(model.state_dict(), strict=True)
            x = torch.randn(2, 13, 32, device="cuda", dtype=dtype, requires_grad=True)
            other = x.detach().clone().requires_grad_(True)
            out, (_, selected) = model(x)
            expected, (_, ref_selected) = reference(other)
            assert torch.equal(selected.reshape_as(ref_selected), ref_selected)
            d_output = torch.randn_like(out)
            out.backward(d_output)
            expected.backward(d_output)
            actual_grad, expected_grad = [], []
            refs = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None
                actual_grad.append(parameter.grad.flatten())
                expected_grad.append(refs[name].grad.flatten())
            result = {"dtype": str(dtype), "top_k": top_k, "output": error(out, expected),
                      "input_gradient": error(x.grad, other.grad),
                      "parameter_gradient": error(torch.cat(actual_grad), torch.cat(expected_grad))}
            tolerance = 2e-5 if dtype == torch.float32 else .01
            assert all(result[key]["relative_l2"] < tolerance
                       for key in ("output", "input_gradient", "parameter_gradient")), result
            rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                                     "reference_revision": "fcc60a52b34c69c6271d0ac8a8cb0815020b6a43",
                                     "cases": rows}, indent=2) + "\n")
    print(f"Official Ling GPU FFN forward/backward passed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
