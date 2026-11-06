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

import os
import threading
import time
from abc import ABC
from typing import Any, Dict, Optional, Sequence

import torch
from mooncake.store import MooncakeDistributedStore

from aspec.config.mooncake_config import MooncakeConfig
from aspec.transfer.mooncake.buffers import (
    AsyncPutManager,
    GPUReceiveBuffer,
    GPUSendBuffer,
    HostBufferPool,
)
from aspec.utils.logging import logger


class MooncakeHiddenStateStore(ABC):
    """
    Base class for Mooncake Store wrapper to store hidden states from target model.

    Uses RDMA-registered host buffers (MooncakeHostMemAllocator) and put_from
    for zero-copy transfers. Optionally uses GPU Direct RDMA for both sending
    and receiving.
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._init_event = threading.Event()
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None
        self._async_put_manager: Optional[AsyncPutManager] = None
        self._gpu_receive_buffer: Optional[GPUReceiveBuffer] = None
        self._gpu_send_buffer: Optional[GPUSendBuffer] = None
        self._gpu_direct_available = False
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._replicate_config: Any = None
        self._p2p_transfer = None

    def setup(
        self,
        device: torch.device | int | None = None,
        *,
        p2p_role: str | None = None,
    ) -> None:
        """Initialize the Mooncake Store client."""
        if self._initialized:
            return
        # A failed initialization can still own registered memory. Cleanup must
        # succeed before replacing the client or any of its allocation owners.
        if self._store is not None or self._p2p_transfer is not None:
            self.close()

        if device is not None and not isinstance(device, torch.device):
            device = torch.device("cuda", int(device))

        if self.config.protocol == "rdma" and not self.config.device_name:
            logger.warning(
                "RDMA protocol with empty device_name will use auto-discovery, "
                "which may fail on hosts with mixed IB subnets. "
                "Set mooncake.device_name to a specific RDMA device (e.g. 'mlx5_0')."
            )

        self.config.apply_env_defaults()
        # The driver-built config may carry the driver's address. Resolve it
        # again inside a Ray actor, but preserve an explicitly propagated
        # address inside backend child processes. Those children are not Ray
        # workers, and generic default-route discovery can select a different
        # interface from the preflight-validated Ray cluster interface.
        try:
            import ray

            from aspec.ray.ray_actor import RayActor

            if ray.is_initialized():
                local_ip = RayActor.get_node_ip()
                if local_ip and local_ip != self.config.local_hostname:
                    logger.info(
                        "Overriding mooncake local_hostname %s -> %s "
                        "(Ray NodeManager address)",
                        self.config.local_hostname,
                        local_ip,
                    )
                    self.config.local_hostname = local_ip
                    os.environ["MOONCAKE_LOCAL_HOSTNAME"] = local_ip
        except Exception:
            logger.warning(
                "Could not re-resolve local node IP for mooncake; using configured value"
            )

        # Proxies inherited from login shells must not intercept Mooncake's
        # intra-cluster metadata traffic.
        for variable in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        ):
            if os.environ.pop(variable, None):
                logger.info("Unset %s for mooncake transfer engine", variable)

        if self.config.data_plane == "transfer_engine":
            if p2p_role is None:
                raise ValueError(
                    "transfer_engine data_plane requires setup(..., "
                    "p2p_role='source'|'destination')"
                )
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            from aspec.transfer.mooncake.p2p import MooncakeP2PTransfer

            self._p2p_transfer = MooncakeP2PTransfer(self.config)
            self._p2p_transfer.setup(device, role=p2p_role)
            self._initialized = True
            self._init_event.set()
            logger.info(
                "Mooncake TransferEngine data plane initialized "
                "(protocol=%s, role=%s, endpoint=%s)",
                self.config.protocol,
                p2p_role,
                self._p2p_transfer.endpoint,
            )
            return

        if (
            self.config.protocol == "tcp"
            and "MC_TCP_BIND_ADDRESS" not in os.environ
            and self.config.local_hostname
            and self.config.local_hostname != "localhost"
        ):
            os.environ["MC_TCP_BIND_ADDRESS"] = self.config.local_hostname
            logger.info(
                "Set MC_TCP_BIND_ADDRESS=%s for transfer engine",
                self.config.local_hostname,
            )

        self._store = MooncakeDistributedStore()
        logger.info(
            "Connecting to Mooncake master at %s, metadata server at %s",
            self.config.master_server_address,
            self.config.metadata_server,
        )
        result = self._store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result is not None and result != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake client (error={result}). "
                f"Check that Mooncake master is running at {self.config.master_server_address} "
                f"and metadata server is available at {self.config.metadata_server}"
            )

        self._verify_force_delete()
        self._build_replicate_config()

        pool_size = self.config.async_put_pool_size
        if pool_size > 0:
            self._host_buffer_pool = HostBufferPool(
                buffer_size=self.config.host_buffer_size,
                pool_size=pool_size,
                pin_memory=device is None or device.type == "cuda",
            )
            self._host_buffer_pool.initialize()

            for buf in self._host_buffer_pool._buffers:
                if not self._register_buffer(buf.ptr, buf.size):
                    self.close()
                    raise RuntimeError("Failed to register Mooncake host buffer")

            self._async_put_manager = AsyncPutManager(
                store=self._store,
                max_workers=pool_size,
                replicate_config=self._replicate_config,
                store_full_error_codes=self.config.store_full_error_codes,
                store_full_wait_seconds=self.config.store_full_wait_seconds,
                store_full_log_interval_seconds=self.config.store_full_log_interval_seconds,
                store_full_max_wait_seconds=self.config.store_full_max_wait_seconds,
            )
            logger.info("Async put manager created (pool_size=%d)", pool_size)

        if self.config.enable_gpu_direct and torch.cuda.is_available():
            self._setup_gpu_direct(device)

        # Can't create copy stream for "cpu" device (offline replay engine).
        if torch.cuda.is_available() and (device is None or device.type == "cuda"):
            cuda_device = device if device is not None else torch.device("cuda")
            self._copy_stream = torch.cuda.Stream(device=cuda_device)
            logger.info("DtoH copy stream created on %s", cuda_device)

        self._initialized = True
        self._init_event.set()
        logger.info(
            "Mooncake Store client initialized (protocol=%s, device=%s, gpu_direct=%s, pool_size=%d)",
            self.config.protocol,
            self.config.device_name or "(auto-discovery)",
            self._gpu_direct_available,
            pool_size,
        )

    def get_p2p_status(self) -> dict | None:
        """Return TransferEngine data-plane metrics when that path is active."""

        if self._p2p_transfer is None:
            return None
        return self._p2p_transfer.status()

    def _setup_gpu_direct(self, device: torch.device = None) -> None:
        """Initialize GPU send/receive buffers and register for GPU Direct RDMA."""
        try:
            self._gpu_receive_buffer = GPUReceiveBuffer(
                size=self.config.gpu_buffer_size,
                device=device,
            )
            self._gpu_receive_buffer.initialize()

            if not self._register_buffer(
                self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size
            ):
                logger.warning("Failed to register GPU receive buffer with Mooncake")
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
                return

            # Allocate GPU send buffer (same size as host buffer — put is 1 sample)
            self._gpu_send_buffer = GPUSendBuffer(
                size=self.config.host_buffer_size,
                device=device,
            )
            self._gpu_send_buffer.initialize()

            if not self._register_buffer(self._gpu_send_buffer.ptr, self._gpu_send_buffer.size):
                logger.warning("Failed to register GPU send buffer with Mooncake")
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None
                # receive buffer is still usable, continue

            self._gpu_direct_available = True
            send_desc = (
                f"{self._gpu_send_buffer.size / (1024**2):.1f}MB"
                if self._gpu_send_buffer
                else "N/A"
            )
            logger.info(
                "GPU Direct RDMA enabled: receive=%.1fMB, send=%s on %s",
                self._gpu_receive_buffer.size / (1024**2),
                send_desc,
                device or "cuda",
            )

        except Exception as e:
            logger.warning("Failed to setup GPU Direct RDMA: %s", e)
            self._gpu_direct_available = False
            for name in ("_gpu_receive_buffer", "_gpu_send_buffer"):
                buffer = getattr(self, name)
                if buffer is None:
                    continue
                if not self._unregister_buffer(buffer.ptr):
                    raise RuntimeError(
                        "Failed to unregister GPU buffer after initialization failure; "
                        "allocation retained until the Mooncake client closes"
                    ) from e
                buffer.free()
                setattr(self, name, None)

    def _ensure_initialized(self, timeout: float = 30.0) -> None:
        """Block until setup() has completed (thread-safe via Event).

        If setup() was called from a background thread, this will wait up to
        *timeout* seconds for it to finish.  If nobody has called setup() yet,
        falls back to calling it on the current thread.
        """
        if self._init_event.is_set():
            return
        if not self._init_event.wait(timeout=timeout):
            if not self._initialized:
                self.setup()

    def _register_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register a buffer for RDMA transfers."""
        if buffer_ptr in self._registered_buffers:
            return True

        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(buffer_ptr, size)
                if result == 0:
                    self._registered_buffers[buffer_ptr] = size
                    logger.debug("Registered buffer at %#x, size=%s", buffer_ptr, size)
                    return True
                logger.warning("register_buffer returned error code: %s", result)
                return False
        except Exception as e:
            logger.warning("Failed to register buffer: %s", e)

        return False

    @property
    def supports_multi_buffer_put(self) -> bool:
        """Whether this Mooncake client can build one object from many buffers."""
        self._ensure_initialized()
        return callable(getattr(self._store, "batch_put_from_multi_buffers", None))

    def register_external_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register caller-owned memory for zero-copy Mooncake operations.

        The caller retains ownership and must keep the allocation alive until
        the store is closed (or :meth:`unregister_external_buffer` succeeds).
        Sub-ranges of the registered allocation may subsequently be passed to
        ``batch_put_from_multi_buffers``.
        """
        self._ensure_initialized()
        return self._register_buffer(buffer_ptr, size)

    def unregister_external_buffer(self, buffer_ptr: int) -> bool:
        """Unregister caller-owned memory previously registered with Mooncake."""
        self._ensure_initialized()
        return self._unregister_buffer(buffer_ptr)

    def _unregister_buffer(self, buffer_ptr: int) -> bool:
        # Also used while setup is unwinding, before _init_event can be set.
        if buffer_ptr not in self._registered_buffers:
            return True
        try:
            result = self._store.unregister_buffer(buffer_ptr)
        except Exception as exc:
            logger.warning("Failed to unregister buffer at %#x: %s", buffer_ptr, exc)
            return False
        if result not in (None, 0):
            logger.warning("unregister_buffer returned error code %s for %#x", result, buffer_ptr)
            return False
        self._registered_buffers.pop(buffer_ptr, None)
        return True

    def put_from_registered_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> None:
        """Publish objects assembled from registered memory fragments.

        This is the scatter-source counterpart of ``batch_put_from``.  It is
        used by the vLLM PP connector to publish one layer directly from its
        paged hidden-state cache without first packing the blocks into a host
        or GPU staging buffer.
        """
        self._ensure_initialized()
        key_list = list(keys)
        ptr_lists = [list(ptrs) for ptrs in all_buffer_ptrs]
        size_lists = [list(sizes) for sizes in all_sizes]
        if not key_list or len(key_list) != len(ptr_lists) or len(key_list) != len(size_lists):
            raise ValueError(
                "Expected equal non-empty keys/pointer-lists/size-lists, got "
                f"{len(key_list)}/{len(ptr_lists)}/{len(size_lists)}"
            )
        for key, ptrs, sizes in zip(key_list, ptr_lists, size_lists):
            if not ptrs or len(ptrs) != len(sizes):
                raise ValueError(
                    f"Mooncake object {key!r} has {len(ptrs)} pointers and {len(sizes)} sizes"
                )
            if any(size <= 0 for size in sizes):
                raise ValueError(f"Mooncake object {key!r} has a non-positive fragment size")

        batch_put = getattr(self._store, "batch_put_from_multi_buffers", None)
        if not callable(batch_put):
            raise RuntimeError(
                "Mooncake batch_put_from_multi_buffers is unavailable; "
                "mooncake-transfer-engine >= 0.3.12.post1 is required for "
                "direct paged hidden-state publication"
            )
        if self._replicate_config is not None:
            results = list(
                batch_put(key_list, ptr_lists, size_lists, config=self._replicate_config)
            )
        else:
            results = list(batch_put(key_list, ptr_lists, size_lists))
        if len(results) != len(key_list):
            raise RuntimeError(
                "batch_put_from_multi_buffers returned "
                f"{len(results)} results for {len(key_list)} keys"
            )
        failures = [(key, result) for key, result in zip(key_list, results) if result != 0]
        if not failures:
            return
        try:
            self._store.batch_remove(key_list, force=True)
        except Exception:
            logger.warning(
                "Failed to cleanup keys after batch_put_from_multi_buffers failure: %s",
                key_list,
                exc_info=True,
            )
        detail = ", ".join(f"{key} (code={result})" for key, result in failures)
        raise RuntimeError(f"batch_put_from_multi_buffers failed: {detail}")

    def warmup_rdma(self) -> None:
        """Do a small test PUT to warm up the RDMA data path."""
        import uuid

        self._ensure_initialized()
        if self._host_buffer_pool is None:
            logger.debug("Skipping RDMA warmup — no host buffer pool (get-only store)")
            return
        key = f"_warmup_{uuid.uuid4().hex[:8]}"
        buf = self._host_buffer_pool.get_buffer()
        size = 4096
        self._store.batch_put_from([key], [buf.ptr], [size])
        self._store.batch_remove([key], force=True)
        logger.info("RDMA warmup complete")

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store (metadata-only, no data download)."""
        if self.config.data_plane == "transfer_engine":
            from aspec.transfer.mooncake.p2p import is_p2p_key

            return is_p2p_key(key)
        try:
            result = self._store.is_exist(key)
            return result == 1
        except Exception:
            return False

    def batch_exists(self, keys: Sequence[str]) -> Dict[str, bool]:
        """Return a metadata-only existence census for *keys*.

        Newer Mooncake clients provide a one-RPC ``batch_is_exist`` API.  Keep
        a per-key fallback for older supported clients, while preserving
        errors so readers fail closed rather than mistaking a metadata failure
        for a missing object.
        """
        self._ensure_initialized()
        key_list = list(keys)
        batch_is_exist = getattr(self._store, "batch_is_exist", None)
        if callable(batch_is_exist):
            results = list(batch_is_exist(key_list))
            if len(results) != len(key_list):
                raise RuntimeError(
                    "Mooncake batch_is_exist returned "
                    f"{len(results)} results for {len(key_list)} keys"
                )
        else:
            results = [self._store.is_exist(key) for key in key_list]
        return {key: result == 1 for key, result in zip(key_list, results)}

    def wait_for_keys(
        self,
        keys: Sequence[str],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> None:
        """Wait until every key is visible in Mooncake metadata.

        This is deliberately called before either GPUDirect or host-buffer
        byte movement.  A timeout reports the exact missing keys, which turns
        an incomplete pipeline fragment into a fail-closed sample.
        """
        key_list = list(keys)
        if not key_list:
            return
        if timeout is None:
            timeout = self.config.get_retry_max_wait_seconds
        if poll_interval is None:
            poll_interval = self.config.get_retry_wait_seconds
        poll_interval = max(float(poll_interval), 0.001)
        timeout = float(timeout)
        start = time.monotonic()
        deadline = None if timeout <= 0 else start + timeout
        last_missing = key_list

        while True:
            census = self.batch_exists(key_list)
            last_missing = [key for key in key_list if not census[key]]
            if not last_missing:
                return
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                missing = ", ".join(last_missing)
                raise TimeoutError(
                    "Timed out waiting for Mooncake keys after "
                    f"{now - start:.3f}s; missing: {missing}"
                )
            sleep_for = poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(deadline - now, 0.0))
            time.sleep(sleep_for)

    def check_async_errors(self) -> None:
        """Surface completed background put failures without draining puts."""
        self._ensure_initialized()
        if self._async_put_manager is not None:
            self._async_put_manager.check_errors()

    def _verify_force_delete(self) -> None:
        """Fail-fast if Mooncake doesn't support batch_remove(force=True).

        Requires mooncake-transfer-engine >= 0.3.10.post1.
        Primary check uses package version metadata; falls back to docstring
        heuristic for non-pip installs.
        """
        batch_remove = getattr(self._store, "batch_remove", None)
        if batch_remove is None:
            raise RuntimeError(
                "Mooncake version too old: batch_remove() not found. "
                "Requires mooncake-transfer-engine >= 0.3.10.post1."
            )
        try:
            from importlib.metadata import version

            from packaging.version import Version

            installed = Version(version("mooncake-transfer-engine"))
            if installed >= Version("0.3.10.post1"):
                return
        except Exception:
            pass
        doc = getattr(batch_remove, "__doc__", "") or ""
        if "force" not in doc:
            raise RuntimeError(
                "Mooncake version too old: batch_remove(force=True) not supported. "
                "Requires mooncake-transfer-engine >= 0.3.10.post1."
            )

    def _build_replicate_config(self) -> None:
        """Honor requested eviction protection; never silently publish unpinned."""
        self._replicate_config = None
        if not self.config.enable_hard_pin:
            return
        try:
            from mooncake.store import ReplicateConfig
        except ImportError as exc:
            raise RuntimeError(
                "enable_hard_pin=True requires Mooncake ReplicateConfig.with_hard_pin"
            ) from exc
        cfg = ReplicateConfig()
        if not hasattr(cfg, "with_hard_pin"):
            raise RuntimeError(
                "enable_hard_pin=True requires Mooncake ReplicateConfig.with_hard_pin"
            )
        cfg.with_hard_pin = True
        self._replicate_config = cfg
        logger.info("Hard pin enabled for batch_put_from")

    def close(self) -> None:
        """Close the Mooncake Store client."""
        if self._p2p_transfer is not None:
            self._p2p_transfer.close()
            self._p2p_transfer = None
        if self._async_put_manager is not None:
            self._async_put_manager.shutdown()
            self._async_put_manager = None
        # Registered memory must remain alive until the client has stopped
        # using it. On close failure retain allocations so cleanup can retry.
        if self._store is not None:
            self._store.close()
            self._store = None
        self._registered_buffers.clear()
        if self._gpu_send_buffer is not None:
            self._gpu_send_buffer.free()
            self._gpu_send_buffer = None
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        self._copy_stream = None
        self._initialized = False
        self._init_event.clear()
        self._gpu_direct_available = False
