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

import json
from pathlib import Path

import pytest

from aspec.config.train_config import load_config
from aspec.data.parse import GeneralParser, create_parser
from aspec.data.template import TEMPLATE_REGISTRY
from aspec.models.draft import AutoDraftModelConfig
from aspec.planning import PlanCompileError, _resolve_target_family, compile_execution_plan

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/recipes/ling3_tiny_dspark.yaml"


def test_ling_recipe_resolves_explicit_feature_contract():
    config = load_config(str(RECIPE))
    assert config.dataset.train_data_path == "/tmp/aspec-test/train.parquet"
    config.model.draft_model_config = str(
        ROOT / "aspec/config/dspark_ling3_tiny_draft_config.json"
    )
    plan = compile_execution_plan(config).to_dict()
    assert plan["workload"]["models"]["target"]["family"] == "ling3"
    features = plan["workload"]["features"]
    assert features["target_hidden_size"] == 1536
    assert features["target_num_hidden_layers"] == 24
    assert features["aux_hidden_state_layers"] == [1, 6, 11, 16, 21]
    assert features["tensors"]["hidden_states"]["shape"] == ["sequence", 7680]
    assert features["tensors"]["last_hidden_states"]["value_stage"] == "post_norm"


def test_ling_draft_reuses_the_exportable_dense_dspark_config():
    config = AutoDraftModelConfig.from_file(str(
        ROOT / "aspec/config/dspark_ling3_tiny_draft_config.json"
    ))
    # convert_to_hf already maps this dense backbone to Transformers' qwen3
    # config; an unregistered "dspark" model_type cannot be consumed upstream.
    assert config.model_type == "qwen3_dspark"
    assert config.architectures == ["DSparkDraftModel"]
    assert config.hidden_size == 1536
    assert config.vocab_size == 157184


def test_ling_model_identity_uses_checkpoint_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "bailing_hybrid", "architectures": ["BailingMoeV3ForCausalLM"]}
        )
    )
    assert _resolve_target_family(str(tmp_path)) == "ling3"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unrelated", "architectures": ["UnrelatedModel"]})
    )
    with pytest.raises(PlanCompileError, match="supported target scope"):
        _resolve_target_family(str(tmp_path))


def test_ling_template_preserves_reasoning_and_tool_messages():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "formatted"

    tokenizer = Tokenizer()
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("ling3"))
    assert type(parser) is GeneralParser
    messages = [
        {"role": "user", "content": "Find the result."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the tool.",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}],
        },
        {"role": "tool", "content": "42"},
        {"role": "assistant", "content": "42", "reasoning_content": "Now answer."},
    ]
    assert parser.format(messages) == "formatted"
    assert tokenizer.messages == messages
