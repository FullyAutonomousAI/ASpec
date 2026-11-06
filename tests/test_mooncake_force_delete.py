# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Tests for Mooncake Store force delete + hard pin refactoring.

Depends on conftest.py (project root) to stub torch and mooncake when
running on environments without GPU dependencies (e.g. Mac dev machines).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from aspec.config.mooncake_config import MooncakeConfig
from aspec.transfer.mooncake.buffers import AsyncPutManager
from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from aspec.transfer.mooncake.store import MooncakeHiddenStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_eagle_store(mock_raw_store):
    """Create an EagleMooncakeStore with a mocked internal _store."""
    config = MooncakeConfig()
    store = EagleMooncakeStore(config)
    store._store = mock_raw_store
    store._initialized = True
    store._init_event.set()
    return store


def _make_base_store(mock_raw_store, enable_hard_pin=False):
    """Create a MooncakeHiddenStateStore subclass with a mocked internal _store."""
    config = MooncakeConfig(enable_hard_pin=enable_hard_pin)

    class ConcreteStore(MooncakeHiddenStateStore):
        pass

    store = ConcreteStore(config)
    store._store = mock_raw_store
    return store


# ---------------------------------------------------------------------------
# Test 1: enable_hard_pin env roundtrip
# ---------------------------------------------------------------------------
class TestEnableHardPinConfig:
    def test_enable_hard_pin_env_roundtrip(self):
        config = MooncakeConfig(enable_hard_pin=True)
        assert config.enable_hard_pin is True

        with patch.dict(os.environ, {}, clear=False):
            config.export_env()
            assert os.environ["MOONCAKE_ENABLE_HARD_PIN"] == "1"
            restored = MooncakeConfig.from_env()
            assert restored.enable_hard_pin is True

    def test_enable_hard_pin_default_off(self):
        config = MooncakeConfig()
        assert config.enable_hard_pin is False

        with patch.dict(os.environ, {}, clear=False):
            config.export_env()
            assert os.environ["MOONCAKE_ENABLE_HARD_PIN"] == "0"
            restored = MooncakeConfig.from_env()
            assert restored.enable_hard_pin is False


class TestMooncakeEnvDefaults:
    def test_tcp_memcpy_default_is_applied_by_export_env(self):
        config = MooncakeConfig(protocol="tcp")

        with patch.dict(os.environ, {}, clear=True):
            config.export_env()

            assert os.environ["MC_STORE_MEMCPY"] == "0"
            assert os.environ["MC_FORCE_TCP"] == "1"

    def test_tcp_memcpy_default_preserves_user_override(self):
        config = MooncakeConfig(protocol="tcp")

        with patch.dict(os.environ, {"MC_STORE_MEMCPY": "1"}, clear=True):
            config.apply_env_defaults()

            assert os.environ["MC_STORE_MEMCPY"] == "1"

    def test_tcp_memcpy_default_not_applied_for_rdma(self):
        config = MooncakeConfig(protocol="rdma")

        with patch.dict(os.environ, {}, clear=True):
            config.apply_env_defaults()

            assert "MC_STORE_MEMCPY" not in os.environ
            assert "MC_FORCE_TCP" not in os.environ

    def test_required_rdma_transfer_engine_forces_hca(self):
        config = MooncakeConfig(
            protocol="rdma",
            data_plane="transfer_engine",
            enable_gpu_direct=True,
            allow_host_fallback=False,
        )

        with patch.dict(os.environ, {}, clear=True):
            config.apply_env_defaults()

            assert os.environ["MC_FORCE_HCA"] == "1"

    def test_direct_store_setup_applies_tcp_memcpy_before_mooncake_client_setup(self):
        config = MooncakeConfig(protocol="tcp", async_put_pool_size=0)
        mock_raw_store = MagicMock()
        mock_raw_store.setup.return_value = 0

        class ConcreteStore(MooncakeHiddenStateStore):
            pass

        def make_raw_store():
            assert os.environ["MC_STORE_MEMCPY"] == "0"
            return mock_raw_store

        store = ConcreteStore(config)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("aspec.transfer.mooncake.store.MooncakeDistributedStore", make_raw_store),
            patch.object(ConcreteStore, "_verify_force_delete"),
            patch.object(ConcreteStore, "_build_replicate_config"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            store.setup()

        mock_raw_store.setup.assert_called_once()


# ---------------------------------------------------------------------------
# Tests 2-3: _verify_force_delete
# ---------------------------------------------------------------------------
class TestVerifyForceDelete:
    def test_missing_batch_remove_method(self):
        mock_raw = MagicMock(spec=[])  # no batch_remove attr
        store = _make_base_store(mock_raw)
        with pytest.raises(RuntimeError, match="batch_remove.*not found"):
            store._verify_force_delete()

    def test_missing_force_param(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove = MagicMock()
        mock_raw.batch_remove.__doc__ = "batch_remove(keys) -> list[int]"
        store = _make_base_store(mock_raw)
        # Patch importlib.metadata.version to force fallback to docstring check
        with patch.dict(
            sys.modules, {"importlib.metadata": MagicMock(version=MagicMock(side_effect=Exception))}
        ):
            with pytest.raises(RuntimeError, match="batch_remove.*force.*not supported"):
                store._verify_force_delete()

    def test_valid_batch_remove(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove = MagicMock()
        mock_raw.batch_remove.__doc__ = "batch_remove(keys, force=False) -> list[int]"
        store = _make_base_store(mock_raw)
        store._verify_force_delete()  # should not raise


# ---------------------------------------------------------------------------
# Tests 4-9: remove_eagle3_tensors
# ---------------------------------------------------------------------------
class TestRemoveEagle3Tensors:
    @pytest.fixture(autouse=True)
    def _skip_retry_delays(self):
        with patch("aspec.transfer.mooncake.eagle_store.time.sleep"):
            yield

    def test_success(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        mock_raw.batch_remove.assert_called_once_with(["k1_hs", "k1_ids"], force=True)

    def test_not_found_is_success(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [-704, 0]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 1

    def test_retry_then_succeed(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = [
            [-1, 0],  # first: k1_hs fails
            [0],  # second: k1_hs succeeds
        ]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 2

    def test_exhaust_retries(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [-1, -1]
        store = _make_eagle_store(mock_raw)
        # Should not raise
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 3

    def test_exception_is_retried(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = [
            RuntimeError("connection lost"),
            [0, 0],
        ]
        store = _make_eagle_store(mock_raw)
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 2

    def test_all_exceptions(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.side_effect = RuntimeError("down")
        store = _make_eagle_store(mock_raw)
        # Should not raise
        store.remove_eagle3_tensors("k1")
        assert mock_raw.batch_remove.call_count == 3


class TestDiscardUnconsumedEagle3Tensors:
    def test_store_data_plane_uses_force_delete(self):
        mock_raw = MagicMock()
        mock_raw.batch_remove.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)

        store.discard_unconsumed_eagle3_tensors("k1")

        mock_raw.batch_remove.assert_called_once_with(["k1_hs", "k1_ids"], force=True)

    def test_transfer_engine_reads_and_releases_every_cp_descriptor(self):
        from aspec.transfer.mooncake.p2p import (
            MooncakeP2PDescriptor,
            MooncakeP2PTensorDescriptor,
            pack_direct_cp_p2p_keys,
        )

        config = MooncakeConfig(
            data_plane="transfer_engine",
            protocol="nvlink_intra",
            enable_gpu_direct=True,
            allow_host_fallback=False,
        )
        store = EagleMooncakeStore(config)
        store._initialized = True
        store._p2p_transfer = MagicMock()
        leases = [MagicMock(), MagicMock()]
        store._p2p_transfer.read.side_effect = [
            MagicMock(lease=lease) for lease in leases
        ]
        descriptors = [
            MooncakeP2PDescriptor(
                version=3,
                run_id="run",
                attempt_id="attempt",
                store_key=f"sample_cp{rank}",
                data_id="sample",
                request_id="request",
                source_engine="0",
                source_endpoint="10.0.0.1:15001",
                source_release_endpoint="10.0.0.1:16001",
                slot_base_ptr=0x100000,
                slot_id=rank,
                generation=1,
                payload_bytes=16,
                tensors=(
                    MooncakeP2PTensorDescriptor(
                        "hidden_states", (1, 8), "bfloat16", 0, 16
                    ),
                ),
            )
            for rank in range(2)
        ]
        key = pack_direct_cp_p2p_keys(
            descriptor.to_key() for descriptor in descriptors
        )

        store.discard_unconsumed_eagle3_tensors(key)

        assert [call.args[0] for call in store._p2p_transfer.read.call_args_list] == descriptors
        for lease in leases:
            lease.release.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests 10-11: cleanup does not mask put error
# ---------------------------------------------------------------------------
class TestCleanupDoesNotMaskPutError:
    def test_sync_put_cleanup_does_not_mask_put_error(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [-1, 0]
        mock_raw.batch_remove.side_effect = RuntimeError("cleanup failed")
        store = _make_eagle_store(mock_raw)
        store._replicate_config = None

        with pytest.raises(RuntimeError, match="batch_put_from failed"):
            store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])

    def test_async_put_cleanup_does_not_mask_put_error(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [-1, 0]
        mock_raw.batch_remove.side_effect = RuntimeError("cleanup failed")
        mgr = AsyncPutManager(store=mock_raw, max_workers=1)
        with pytest.raises(RuntimeError, match="async batch_put_from failed"):
            mgr._do_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Tests 12-13: _build_replicate_config
# ---------------------------------------------------------------------------
class TestBuildReplicateConfig:
    def test_supported(self):
        mock_config_instance = MagicMock()
        mock_config_instance.with_hard_pin = False
        mock_store_module = MagicMock()
        mock_store_module.ReplicateConfig.return_value = mock_config_instance

        with patch.dict(sys.modules, {"mooncake.store": mock_store_module}):
            mock_raw = MagicMock()
            store = _make_base_store(mock_raw, enable_hard_pin=True)
            store._build_replicate_config()
            assert store._replicate_config is not None
            assert store._replicate_config.with_hard_pin is True

    def test_unsupported(self):
        mock_config_instance = MagicMock(spec=[])  # no with_hard_pin attr
        mock_store_module = MagicMock()
        mock_store_module.ReplicateConfig.return_value = mock_config_instance

        with patch.dict(sys.modules, {"mooncake.store": mock_store_module}):
            mock_raw = MagicMock()
            store = _make_base_store(mock_raw, enable_hard_pin=True)
            with pytest.raises(RuntimeError, match="requires.*with_hard_pin"):
                store._build_replicate_config()

    def test_missing_config_fails_closed(self):
        with patch.dict(sys.modules, {"mooncake.store": None}):
            store = _make_base_store(MagicMock(), enable_hard_pin=True)
            with pytest.raises(RuntimeError, match="requires.*with_hard_pin"):
                store._build_replicate_config()


# ---------------------------------------------------------------------------
# Tests 14-15: replicate_config passed through put paths
# ---------------------------------------------------------------------------
class TestReplicateConfigPassthrough:
    def test_sync_put_passes_replicate_config(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        mock_cfg = MagicMock()
        store._replicate_config = mock_cfg

        store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(
            ["k_hs", "k_ids"], [100, 200], [64, 32], config=mock_cfg
        )

    def test_sync_put_no_config_when_none(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        store = _make_eagle_store(mock_raw)
        store._replicate_config = None

        store._do_sync_batch_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(["k_hs", "k_ids"], [100, 200], [64, 32])

    def test_async_put_passes_replicate_config(self):
        mock_raw = MagicMock()
        mock_raw.batch_put_from.return_value = [0, 0]
        mock_cfg = MagicMock()
        mgr = AsyncPutManager(store=mock_raw, max_workers=1, replicate_config=mock_cfg)
        mgr._do_put(["k_hs", "k_ids"], [100, 200], [64, 32])
        mock_raw.batch_put_from.assert_called_once_with(
            ["k_hs", "k_ids"], [100, 200], [64, 32], config=mock_cfg
        )
        mgr.shutdown()


class TestPinnedStoreBackpressure:
    def test_capacity_retry_preserves_pin_and_cleans_partial_batch(self):
        raw, config = MagicMock(), MagicMock()
        raw.batch_put_from.side_effect = [[-200, 0], [0, 0]]
        mgr = AsyncPutManager(raw, replicate_config=config)
        with patch("aspec.transfer.mooncake.buffers.time.sleep") as sleep:
            mgr._do_put(["hs", "ids"], [100, 200], [64, 32])
        assert raw.batch_put_from.call_count == 2
        assert all(call.kwargs["config"] is config for call in raw.batch_put_from.call_args_list)
        raw.batch_remove.assert_called_once_with(["hs", "ids"], force=True)
        sleep.assert_called_once_with(0.5)
        mgr.shutdown()

    def test_non_capacity_error_fails_without_retry(self):
        raw = MagicMock()
        raw.batch_put_from.return_value = [-1]
        mgr = AsyncPutManager(raw)
        with patch("aspec.transfer.mooncake.buffers.time.sleep") as sleep:
            with pytest.raises(RuntimeError, match="code=-1"):
                mgr._do_put(["hs"], [100], [64])
        sleep.assert_not_called()
        raw.batch_remove.assert_called_once_with(["hs"], force=True)
        mgr.shutdown()

    def test_capacity_wait_honors_explicit_deadline_without_real_sleep(self):
        raw = MagicMock()
        raw.batch_put_from.return_value = [-200]
        mgr = AsyncPutManager(raw, store_full_max_wait_seconds=30)
        with (
            patch("aspec.transfer.mooncake.buffers.time.monotonic", side_effect=[0, 31]),
            patch("aspec.transfer.mooncake.buffers.time.sleep") as sleep,
        ):
            with pytest.raises(RuntimeError, match="code=-200"):
                mgr._do_put(["hs"], [100], [64])
        sleep.assert_not_called()
        mgr.shutdown()

class TestDiscardUnpublishedStoreObjects:
    @pytest.fixture(autouse=True)
    def _skip_retry_delays(self):
        with patch("aspec.transfer.mooncake.eagle_store.time.sleep"):
            yield

    def test_drains_async_writes_before_delete_even_after_an_earlier_error(self):
        events = []
        raw = MagicMock()
        raw.batch_remove.side_effect = lambda keys, **kwargs: events.append("delete") or [0] * len(keys)
        store = _make_eagle_store(raw)
        manager = AsyncPutManager(raw)
        store._async_put_manager = manager
        manager._last_error = RuntimeError("earlier put failure")
        future = MagicMock()
        future.result.side_effect = lambda: events.append("pending write finished")
        manager._in_flight[1] = future
        try:
            store.discard_unpublished_objects(["row_hs", "row_ids", "row_lhs"])
            assert events == ["pending write finished", "delete"]
            assert manager._in_flight == {}
            raw.batch_remove.assert_called_once_with(["row_hs", "row_ids", "row_lhs"], force=True)
            with pytest.raises(RuntimeError, match="earlier put failure"):
                manager.check_last_error()
        finally:
            manager.shutdown()

    def test_retry_only_failed_objects_and_accept_not_found(self):
        raw = MagicMock()
        raw.batch_remove.side_effect = [[0, -1, -704], [0]]
        store = _make_eagle_store(raw)
        store.discard_unpublished_objects(["row_hs", "row_ids", "row_lhs"])
        assert raw.batch_remove.call_args_list[1].args == (["row_ids"],)

    @pytest.mark.parametrize("response", [RuntimeError("down"), [-1, -1], [0]])
    def test_unconfirmed_delete_is_not_silently_accepted(self, response):
        raw = MagicMock()
        if isinstance(response, Exception):
            raw.batch_remove.side_effect = response
        else:
            raw.batch_remove.return_value = response
        with pytest.raises(RuntimeError, match="unpublished"):
            _make_eagle_store(raw).discard_unpublished_objects(["row_hs", "row_ids"])
        assert raw.batch_remove.call_count == 3

    def test_transfer_engine_objects_cannot_be_force_deleted(self):
        store = _make_eagle_store(MagicMock())
        store.config.data_plane = "transfer_engine"
        with pytest.raises(RuntimeError, match="Store"):
            store.discard_unpublished_objects(["row_hs"])
        store._store.batch_remove.assert_not_called()
