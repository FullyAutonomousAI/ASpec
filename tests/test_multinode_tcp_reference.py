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

from aspec.config.train_config import load_config
from aspec.planning import compile_execution_plan


def test_two_node_tcp_reference_uses_normal_configurable_pipeline(tmp_path, monkeypatch):
    repo = Path(__file__).parents[1]
    reference = repo / 'configs/examples/qwen3_8b_4k_multinode_tcp.yaml'
    target = tmp_path / 'Qwen3-8B'
    target.mkdir()
    (target / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3',
        'architectures': ['Qwen3ForCausalLM'],
        'max_position_embeddings': 32768,
        'hidden_size': 4096,
        'num_hidden_layers': 36,
    }))
    draft = tmp_path / 'draft.json'
    draft.write_bytes((repo / 'aspec/config/dspark_draft_config.json').read_bytes())
    monkeypatch.setenv('ASPEC_SHARED_ROOT', str(tmp_path))
    monkeypatch.setenv('ASPEC_TRAINING_NODE_IP', '10.0.0.1')
    monkeypatch.setenv('ASPEC_INFERENCE_NODE_IP', '10.0.0.2')
    monkeypatch.setenv('ASPEC_TRAIN_DATA', str(tmp_path / 'train.parquet'))
    monkeypatch.setenv('ASPEC_EVAL_DATA', str(tmp_path / 'eval.parquet'))
    config = load_config(config_path=str(reference), cli_args=[
        f'model.target_model_path={target}',
        f'model.draft_model_config={draft}',
    ])
    plan = compile_execution_plan(config, config_path=str(reference)).to_dict()
    assert config.cluster.mode == 'static_multi_node'
    assert config.training.training_node_ips == ['10.0.0.1']
    assert config.inference.inference_node_ips == ['10.0.0.2']
    assert config.training.train_env_vars == '{}'
    assert config.mooncake['enable_hard_pin'] is True
    assert config.training.training_num_gpus_per_node == 8
    assert config.inference.inference_num_gpus == 8
    assert plan['policy']['pipeline']['global_batch_size'] == 512
    assert plan['policy']['pipeline']['streaming_dispatch_with_training'] is True
    assert plan['policy']['pipeline']['pipeline_dispatch_with_training'] is True
    transfer = plan['policy']['capacity']['transfer_storage']
    assert transfer['data_plane'] == 'store' and transfer['protocol'] == 'tcp'
    assert transfer['host_staging_selected'] is True
    assert transfer['host_fallback_allowed'] is False
    assert transfer['producer_pinned_host_bytes_per_endpoint'] == 1024**3
    assert transfer['local_buffer_size'] == '8GB'
    assert transfer['global_segment_size'] == '32GB'
    assert plan['fallbacks'] == []


def test_node_local_reference_does_not_require_shared_mount_or_fixed_interface(tmp_path, monkeypatch):
    repo = Path(__file__).parents[1]
    reference = repo / 'configs/recipes/qwen3_8b_4k_multinode_node_local_nvlink.yaml'
    target = tmp_path / 'Qwen3-8B'
    target.mkdir()
    (target / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3',
        'architectures': ['Qwen3ForCausalLM'],
        'max_position_embeddings': 32768,
        'hidden_size': 4096,
        'num_hidden_layers': 36,
    }))
    draft = tmp_path / 'draft.json'
    draft.write_bytes((repo / 'aspec/config/dspark_draft_config.json').read_bytes())
    monkeypatch.setenv('ASPEC_SHARED_ROOT', str(tmp_path))
    monkeypatch.setenv('ASPEC_REPO_ROOT', str(repo))
    monkeypatch.setenv('ASPEC_NODE_0_IP', '10.0.0.1')
    monkeypatch.setenv('ASPEC_NODE_1_IP', '10.0.0.2')
    monkeypatch.setenv('ASPEC_TRAIN_DATA', str(tmp_path / 'train.parquet'))
    monkeypatch.setenv('ASPEC_EVAL_DATA', str(tmp_path / 'eval.parquet'))
    config = load_config(config_path=str(reference), cli_args=[
        f'model.target_model_path={target}',
        f'model.draft_model_config={draft}',
    ])

    train_env = json.loads(config.training.train_env_vars)
    assert train_env == {}
    assert config.cluster.storage_mode == "node_local"
    assert config.cluster.shared_root is None
    assert config.training.training_node_ips == ['10.0.0.1', '10.0.0.2']
    assert config.inference.inference_node_ips == ['10.0.0.1', '10.0.0.2']
    assert config.mooncake['feature_affinity'] == 'node_local_required'


def test_tcp_control_long_context_overrides_keep_replicated_eight_plus_eight(tmp_path, monkeypatch):
    repo = Path(__file__).parents[1]
    reference = repo / 'configs/examples/qwen3_8b_4k_multinode_tcp.yaml'
    target = tmp_path / 'Qwen3-8B-YARN'
    target.mkdir()
    (target / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3',
        'architectures': ['Qwen3ForCausalLM'],
        'max_position_embeddings': 131072,
        'hidden_size': 4096,
        'num_hidden_layers': 36,
        'rope_scaling': {
            'rope_type': 'yarn',
            'factor': 8.0,
            'original_max_position_embeddings': 32768,
        },
    }))
    draft = tmp_path / 'draft.json'
    draft.write_bytes((repo / 'aspec/config/dspark_draft_config.json').read_bytes())
    monkeypatch.setenv('ASPEC_TARGET_MODEL', str(target))
    monkeypatch.setenv('ASPEC_DRAFT_CONFIG', str(draft))
    monkeypatch.setenv('ASPEC_SHARED_ROOT', str(tmp_path))
    monkeypatch.setenv('ASPEC_TRAINING_NODE_IP', '10.0.0.1')
    monkeypatch.setenv('ASPEC_INFERENCE_NODE_IP', '10.0.0.2')
    monkeypatch.setenv('ASPEC_TRAIN_DATA', str(tmp_path / 'train.parquet'))
    monkeypatch.setenv('ASPEC_EVAL_DATA', str(tmp_path / 'eval.parquet'))

    config = load_config(config_path=str(reference), cli_args=[
        f'model.target_model_path={target}',
        f'model.draft_model_config={draft}',
        'training.max_seq_length=131072',
        'training.draft_accumulation_steps=8',
        'inference.inference_buffer_threshold=32',
        'inference.max_sample_pool_size=64',
        'inference.inference_batch_size=1',
        'inference.sglang.mem_fraction_static=0.65',
        'inference.sglang.allow_context_length_override=true',
        'inference.sglang.extra_args.max_running_requests=1',
        'inference.sglang.extra_args.max_total_tokens=131328',
        'inference.sglang.extra_args.max_prefill_tokens=131328',
        'mooncake.max_seq_len=131072',
        'mooncake.global_segment_size=40GB',
        'mooncake.local_buffer_size=24GB',
        'mooncake.host_buffer_size=24GB',
    ])
    plan = compile_execution_plan(config, config_path=str(reference)).to_dict()

    assert config.training.fsdp_strategy == 'REPLICATE'
    assert config.training.attention_backend == 'cudnn_flex_attention'
    assert config.training.training_num_gpus_per_node == 8
    assert config.inference.inference_num_gpus == 8
    assert config.inference.inference_node_ips == ['10.0.0.2']
    assert plan['policy']['pipeline']['global_batch_size'] == 64
    assert (
        plan['policy']['capacity']['host_orchestrator']['max_sample_pool_size']
        == 64
    )
    transfer = plan['policy']['capacity']['transfer_storage']
    assert transfer['data_plane'] == 'store'
    assert transfer['protocol'] == 'tcp'
    assert transfer['host_staging_selected'] is True
