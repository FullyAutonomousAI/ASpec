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

import copy

import pytest
import torch

from aspec.feature_contract import (
    FeatureContractError,
    FeatureContractRuntime,
    build_target_feature_contract,
    validate_feature_contract,
)


def _contract():
    return build_target_feature_contract(
        drafter_family="dspark",
        target_hidden_size=4,
        target_num_hidden_layers=12,
        aux_hidden_state_layers=[1, 5, 9],
        store_last_hidden_states=True,
    )


def _metadata(contract, sequence=7):
    return {
        "tensor_shapes": {
            "hidden_states": (sequence, 12),
            "input_ids": (sequence,),
            "last_hidden_states": (sequence, 4),
        },
        "tensor_dtypes": {
            "hidden_states": torch.bfloat16,
            "input_ids": torch.int64,
            "last_hidden_states": torch.bfloat16,
        },
        "metadata": {"aspec_feature_contract_id": contract["contract_id"]},
    }


def test_contract_is_deterministic_and_describes_actual_dspark_boundary():
    first = _contract()
    second = _contract()

    assert first == second
    assert first["producer"] == {"backend": "sglang", "mode": "prefill_only"}
    assert first["consumer"]["target_distribution"] == (
        "consumer_lm_head_from_last_hidden_states"
    )
    assert first["tensors"]["hidden_states"]["shape"] == ["sequence", 12]
    assert first["tensors"]["last_hidden_states"]["value_stage"] == "post_norm"
    assert first["omissions"] == [
        {"tensor": "logits", "reason": "computed_by_consumer_lm_head"}
    ]


def test_mutated_contract_identity_is_rejected():
    contract = _contract()
    contract["aux_hidden_state_layers"] = [1, 6, 9]

    with pytest.raises(FeatureContractError, match="identity mismatch"):
        validate_feature_contract(contract)


def test_producer_and_consumer_validate_the_same_contract():
    contract = _contract()
    producer = FeatureContractRuntime(contract, role="producer", participant="engine_0")
    consumer = FeatureContractRuntime(contract, role="consumer", participant="trainer_0")
    observed = _metadata(contract)

    producer.validate_producer_binding(
        hidden_size=4,
        num_hidden_layers=12,
        aux_hidden_state_layers=[1, 5, 9],
        store_last_hidden_states=True,
    )
    producer.validate_metadata(**observed)
    consumer.validate_metadata(**observed)
    consumer.validate_tensors(
        {
            "hidden_states": torch.zeros((7, 12), dtype=torch.bfloat16),
            "input_ids": torch.zeros((7,), dtype=torch.int64),
            "last_hidden_states": torch.zeros((7, 4), dtype=torch.bfloat16),
        },
        expected_shapes=observed["tensor_shapes"],
        expected_dtypes=observed["tensor_dtypes"],
    )

    assert producer.status()["state"] == "validated"
    assert consumer.status()["state"] == "validated"
    assert consumer.status()["tensor_validations"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["metadata"].update(aspec_feature_contract_id="wrong"), "identity"),
        (lambda value: value["tensor_shapes"].update(hidden_states=(7, 8)), "shape"),
        (lambda value: value["tensor_dtypes"].update(input_ids=torch.int32), "dtype"),
    ],
)
def test_observed_metadata_mismatch_fails_closed(mutation, message):
    contract = _contract()
    observed = _metadata(contract)
    mutation(observed)
    runtime = FeatureContractRuntime(contract, role="consumer", participant="trainer_0")

    with pytest.raises(FeatureContractError, match=message):
        runtime.validate_metadata(**observed)


def test_invalid_layer_contract_fails_before_backend_start():
    with pytest.raises(FeatureContractError, match="unique"):
        build_target_feature_contract(
            drafter_family="dspark",
            target_hidden_size=4,
            target_num_hidden_layers=12,
            aux_hidden_state_layers=[5, 1, 5],
            store_last_hidden_states=True,
        )

    mutated = copy.deepcopy(_contract())
    mutated["contract_id"] = None
    with pytest.raises(FeatureContractError, match="identity mismatch"):
        validate_feature_contract(mutated)
