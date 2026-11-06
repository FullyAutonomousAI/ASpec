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

"""Real CUDA feature/Store probes, not a training or performance benchmark."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from aspec.config.mooncake_config import MooncakeConfig
from aspec.inference.sglang_v0521 import TrainingFeaturePublisher
from aspec.transfer.mooncake import EagleMooncakeStore
from aspec.transfer.mooncake.utils import resolve_mooncake_master_bin


def free_port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    fixture = torch.load(Path(args.fixture) / "features.pt", weights_only=True)
    tokens, captured, layers = (fixture[key] for key in ("input_ids", "captured", "layers"))
    lengths = [len(item) for item in tokens]
    hidden_size = captured[0][1].shape[-1]
    device = torch.device("cuda", 0)
    ids = torch.cat([torch.tensor(item, device=device, dtype=torch.int64) for item in tokens])
    hidden = torch.cat([item[0].to(device) for item in captured])
    teacher = torch.cat([item[1].to(device) for item in captured])
    port, metadata, metrics = free_port(), free_port(), free_port()
    master_log = (root / "master.log").open("w")
    master = subprocess.Popen([
        resolve_mooncake_master_bin(), f"--port={port}",
        f"--http_metadata_server_port={metadata}", "--http_metadata_server_host=127.0.0.1",
        "--enable_http_metadata_server=true", f"--metrics_port={metrics}",
    ], stdout=master_log, stderr=subprocess.STDOUT)
    store = None
    rows = []
    try:
        deadline = time.monotonic() + 30
        while True:
            if master.poll() is not None:
                raise RuntimeError("private Mooncake master exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("private Mooncake master did not start")
                time.sleep(0.1)
        store = EagleMooncakeStore(MooncakeConfig(
            local_hostname="127.0.0.1", master_server_address=f"127.0.0.1:{port}",
            metadata_server=f"http://127.0.0.1:{metadata}/metadata",
            data_plane="store", protocol="tcp", enable_gpu_direct=False, allow_host_fallback=False,
            enable_hard_pin=True, global_segment_size="128MB", local_buffer_size="32MB",
            host_buffer_size="16MB", async_put_pool_size=2, get_retry_max_wait_seconds=15,
        ))
        store.setup(device)
        original_write = store._put_raw_tensors
        original_flush = store.flush
        for scenario in ("handoff", "put_partial", "cp4_partial", "flush_failure", "delete_failure"):
            keys_written = []
            writes = 0

            def write(keys, tensors):
                nonlocal writes
                writes += 1
                keys_written.extend(keys)
                if scenario in {"put_partial", "cp4_partial"} and writes == 2:
                    original_write(keys[:1], tensors[:1])
                    raise RuntimeError("injected partial CUDA/Store put")
                return original_write(keys, tensors)

            def flush():
                original_flush()
                if scenario == "flush_failure":
                    raise RuntimeError("injected CUDA/Store flush failure")

            store._put_raw_tensors, store.flush = write, flush
            cp = 4 if scenario == "cp4_partial" else 1
            if cp > 1:
                os.environ["ASPEC_DIRECT_CP_SHARDED_MOONCAKE"] = "1"
            else:
                os.environ.pop("ASPEC_DIRECT_CP_SHARDED_MOONCAKE", None)
            publisher = TrainingFeaturePublisher({
                "layers": layers, "hidden_size": hidden_size, "context_parallel_size": cp,
                "max_sequence_length": max(lengths), "store_last_hidden_states": True,
                "direct_prefill": False, "attention_backend": "cudnn_flex_attention",
            })
            publisher.store = store
            publisher.capture_layers = tuple(layer + 1 for layer in layers)
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(is_extend=lambda: True),
                is_prefill_only=True, extend_prefix_lens_cpu=[0] * len(tokens),
                extend_seq_lens_cpu=lengths, rids=[f"request-{i}" for i in range(len(tokens))],
                request_custom_params=[
                    {"aspec_data_id": f"{scenario}-{i}", "aspec_request_id": f"request-{i}"}
                    for i in range(len(tokens))
                ],
            )
            injected = None
            try:
                output_keys = publisher.publish(ids, teacher, batch, hidden)
            except RuntimeError as exc:
                if scenario not in {"put_partial", "cp4_partial", "flush_failure"} or "injected" not in str(exc):
                    raise
                injected = str(exc)
                assert publisher.sink is None and not publisher.pending_outputs
            else:
                assert scenario in {"handoff", "delete_failure"}
                if scenario == "delete_failure":
                    raw = store._store

                    class DeleteFault:
                        def __getattr__(self, name):
                            return getattr(raw, name)

                        def batch_remove(self, keys, **kwargs):
                            raise RuntimeError("injected Store delete outage")

                    store._store = DeleteFault()
                    try:
                        publisher.resolve_outputs(batch.rids, [output_keys[1]])
                    except RuntimeError as exc:
                        if "injected Store delete outage" not in str(getattr(exc.__cause__, "__cause__", None)):
                            raise
                        injected = str(exc)
                        assert set(publisher.pending_outputs) == {batch.rids[0]}
                        assert all(store.batch_exists(keys_written).values())
                    else:
                        raise AssertionError("deletion failure was silently accepted")
                    finally:
                        store._store = raw
                    publisher.resolve_outputs(batch.rids, [])
                    kept = 1
                else:
                    publisher.resolve_outputs(batch.rids, [output_keys[0]])
                    kept = 0
                aux, last = captured[kept]
                output = store.get(output_keys[kept], {
                    "input_ids": (lengths[kept],), "hidden_states": tuple(aux.shape),
                    "last_hidden_states": tuple(last.shape),
                }, {"input_ids": torch.int64, "hidden_states": aux.dtype,
                    "last_hidden_states": last.dtype}, device)
                assert torch.equal(output.input_ids.cpu(), torch.tensor(tokens[kept]))
                assert torch.equal(output.hidden_states.cpu(), aux)
                assert torch.equal(output.last_hidden_states.cpu(), last)
                store.remove_eagle3_tensors(output_keys[kept], has_last_hidden_states=True)
                assert not publisher.pending_outputs
            store.flush = original_flush
            assert publisher.drain()
            torch.cuda.synchronize()
            remaining = [key for key, exists in store.batch_exists(keys_written).items() if exists]
            assert not remaining, remaining
            rows.append({
                "scenario": scenario, "injected_error": injected,
                "physical_keys_checked": len(set(keys_written)),
                "remaining_objects": 0, "pending_outputs": len(publisher.pending_outputs),
                "real_async_cuda_to_host_store": True,
                "emitted_sibling_bitwise_equal": scenario in {"handoff", "delete_failure"},
            })
            print(json.dumps(rows[-1]), flush=True)
        (root / "summary.json").write_text(json.dumps({
            "diagnostic_only": True, "fixture": args.fixture, "source": "saved_real_Qwen3_Target_features",
            "device": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
            "protocol": "tcp", "staging": "explicit_host", "cases": rows,
            "claim": "GPU feature/Store lifecycle, not full SGLang cancellation or model-quality acceptance",
        }, indent=2))
    finally:
        if store is not None:
            store.close()
        master.terminate()
        try:
            master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait(timeout=5)
        master_log.close()


if __name__ == "__main__":
    main()
