# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec
import torch
from mooncake.engine import TransferEngine  # type: ignore

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole)
from vllm.config import VllmConfig
from vllm.utils import get_ip, logger, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group)
from vllm_ascend.distributed.mooncake_connector import KVCacheTaskTracker as EncoderCacheTaskTracker, zmq_ctx

if TYPE_CHECKING:
    from vllm.v1.request import Request

class ECMooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    encoder_cache_base_addr: int


class EncoderCacheSendingThread(threading.Thread):

    def __init__(self, tp_rank: int, remote_tp_size: int, local_engine_id: str,
                 side_channel_host: str, side_channel_port: int,
                 metadata: ECMooncakeAgentMetadata,
                 ready_event: threading.Event):
        super().__init__(daemon=True, name="EncoderCacheSendingThread")
        self.tp_rank = tp_rank
        self.remote_tp_size = remote_tp_size
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event

        self.task_tracker = EncoderCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id,
                                                     delay_start_time)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded ECMooncakeAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        # NOTE(rob): we need each rank to have a unique port. This hack to keeps
        # us moving. We will switch when moving to etcd or where we have a
        # single ZMQ socket in the scheduler.
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue

                    msg = decoder.decode(payload[0])
                    if msg[0] == GET_META_MSG:
                        sock.send_multipart((identity, b"", encoded_data))
                    elif msg[0] == DONE_RECVING_MSG:
                        logger.debug("Got DONE_RECVING_MSG for request %s",
                                     msg[1])
                        request_id = msg[1]
                        self.task_tracker.update_done_task_count(request_id)
                        # Acknowledge the request completion.
                        while True:
                            try:
                                # Send ACK to the sender.
                                sock.send_multipart(
                                    (identity, b"", b"ACK"),
                                    flags=zmq.NOBLOCK)  # type: ignore
                                break
                            except zmq.Again:  # type: ignore
                                # If the socket is not ready, retry sending.
                                logger.debug(
                                    "Socket not ready, retrying to send ACK for "
                                    "request %s", msg[1])
                                time.sleep(0.01)
                    else:
                        logger.error(
                            "Connection listener got unexpected message %s",
                            msg)
                except Exception as e:
                    logger.error("Connection listener got exception %s: %s",
                                 type(e), e)


class EncoderCacheRecvingThread(threading.Thread):

    def __init__(self, tp_rank: int, tp_size: int, engine: TransferEngine,
                 local_engine_id: str, local_handshake_port: int,
                 local_ec_cache_base_addr: int, ready_event: threading.Event):
        super().__init__(daemon=True, name="EncoderCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.engine = engine
        self.ready_event = ready_event

        self.encoder_caches_base_addr: dict[str, dict[int, int]] = \
            defaultdict(dict)
        self.encoder_caches_base_addr[local_engine_id][local_handshake_port] = \
            local_ec_cache_base_addr
        self.remote_te_port: dict[str, dict[int, int]] = \
            defaultdict(dict)

        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)

        self.task_tracker = EncoderCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(ECMooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]] = defaultdict(  # type: ignore
                deque)
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

    def add_request(self, request_id: str, local_block_ids: list[int],
                    remote_block_ids: list[int], remote_engine_id: str,
                    remote_host: str, remote_handshake_port: int):
        """Add a new request to the queue for processing."""
        logger.debug(f"Adding request {request_id} to the queue.")
        self.request_queue.put({
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": remote_engine_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
        })

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]

        try:
            logger.debug(
                f"Starting to transfer KV cache for request {request_id}.")
            self._transfer_kv_cache(req_meta)
            logger.debug(
                f"Finished transferring KV cache for request {request_id}.")
        except Exception as e:
            logger.error("Failed to transfer KV cache for request "
                         f"{request_id}: {e}")
        finally:
            self.task_tracker.update_done_task_count(request_id)
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            self._send_done_recv_signal(request_id, remote_host,
                                        remote_handshake_port)
            self.request_queue.task_done()

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        request_id = req_meta["request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            return

        # Check if we have the remote metadata cached.
        if remote_engine_id not in self.kv_caches_base_addr or \
                remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]:
            self._get_remote_metadata(remote_host, remote_handshake_port)

        grouped_remote_block_ids, grouped_local_block_ids = \
            group_concurrent_contiguous(remote_block_ids, local_block_ids)
        remote_kv_caches_base_addrs = \
            self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        local_kv_caches_base_addrs = \
            self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]

        req_start_time = time.perf_counter()
        num_transfer_groups = len(grouped_remote_block_ids)
        num_blocks = len(local_block_ids)

        remote_transfer_port = self.remote_te_port[remote_engine_id][
            remote_handshake_port]
        session_id = f"{remote_host}:{remote_transfer_port}"
        src_list, dst_list, length_list = [], [], []
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)):
            block_len = (self.block_len[k % 2]
                         if self.use_mla else self.block_len[0])
            for i, remote_block_id in enumerate(grouped_remote_block_ids):
                local_block_ids = grouped_local_block_ids[i]
                src = src_layer_base_addr + local_block_ids[0] * block_len
                dst = dst_layer_base_addr + remote_block_id[0] * block_len
                length = len(local_block_ids) * block_len
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)
        ret = self.engine.batch_transfer_sync_read(session_id, src_list,
                                                   dst_list, length_list)
        if ret < 0:
            logger.error("Mooncake transfer failed for request %s",
                         req_meta["request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms (%d groups,"
            " %d blocks).", request_id, req_transfer_elapsed,
            num_transfer_groups, num_blocks)

    def _get_remote_metadata(self, remote_host: str,
                             remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: Optional[zmq.Socket] = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")))
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller)
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id "
                f"{self.local_engine_id}.")
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = \
                agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = \
                agent_meta.te_rpc_port
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host,
                                           remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host,
                             remote_handshake_port)

    def _send_done_recv_signal(self, request_id: str, remote_host: str,
                               remote_handshake_port: int):
        logger.debug("Sending done recving signal for request %s to %s:%d",
                     request_id, remote_host, remote_handshake_port)
        sock: Optional[zmq.Socket] = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id))
            ensure_zmq_send(sock, data_bytes)
            resp = ensure_zmq_recv(sock,
                                   self.remote_poller,
                                   timeout=self.timeout)
            logger.debug(
                f"Received response for request {request_id}: {resp.decode('utf-8')}"
            )
            if resp != b"ACK":
                logger.error("Failed to receive ACK for request %s from %s:%d",
                             request_id, remote_host, remote_handshake_port)
                raise RuntimeError(
                    f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host,
                                           remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host,
                             remote_handshake_port)

    def _get_remote_socket(
            self, remote_host: str,
            remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False)
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000))
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
            self,
            sock: zmq.Socket,  # type: ignore
            remote_host: str,
            remote_handshake_port: int) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


@dataclass
class ECRequestMetadata:
    """Metadata for an encoder cache request"""
    request_id: str
    mm_hashes: List[str]
    remote_engine_id: Optional[str] = None
    remote_host: Optional[str] = None
    remote_port: Optional[int] = None


class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Metadata for EC Mooncake Connector communication between scheduler and worker"""

    def __init__(self, ec_offset: Union[Dict[str, Dict[int, int]], Dict[str, int]]):
        self.requests: Dict[str, ECRequestMetadata] = {}
        self.ec_offset = ec_offset
        
    def add_new_request(
        self,
        request_id: str,
        mm_hashes: List[str],
        ec_transfer_params: dict[str, Any],
    ):
        """Add a new request to the metadata"""
        self.requests[request_id] = ECRequestMetadata(
            request_id=request_id,
            mm_hashes=mm_hashes,
            remote_engine_id=ec_transfer_params.remote_engine_id,
            remote_host=ec_transfer_params.remote_host,
            remote_port=ec_transfer_params.remote_port,
        )


class ECMooncakeConnector(ECConnectorBase):
    """Main EC Mooncake Connector class that handles both scheduler and worker roles"""
    
    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config, role)
        
        self.engine_id = vllm_config.ec_transfer_config.engine_id if hasattr(
            vllm_config, 'ec_transfer_config') else "default_engine"
            
        # Initialize role-specific components
        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler = ECMooncakeConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker = None
        elif role == ECConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = ECMooncakeConnectorWorker(vllm_config, self.engine_id)
        else:
            raise ValueError(f"Unknown role: {role}")
            
        # Cache storage for worker
        self.encoder_caches: Dict[str, torch.Tensor] = {}
        self.cache_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        logger.info(f"Initialized ECMooncakeConnector with role {role.name} and engine_id {self.engine_id}")

    # ==============================
    # Worker-side methods
    # ==============================
    
    def register_cache(self, ec_cache: torch.Tensor):
        assert self.connector_worker is not None
        self.connector_worker.register_cache(ec_cache)

    def start_load_caches(self, **kwargs) -> None:
        """Start loading encoder caches from remote sources"""
        if self.role != ECConnectorRole.WORKER:
            return
            
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ECMooncakeConnectorMetadata):
            logger.error("Invalid metadata type for EC Mooncake Connector")
            return

        encoder_cache = kwargs.get('encoder_cache')

        assert self.connector_worker is not None
        self.connector_worker.start_load_caches(metadata, encoder_cache)

    def save_caches(self, **kwargs) -> None:
        """Save encoder caches to storage/remote destinations"""
        if self.role != ECConnectorRole.WORKER:
            return
            
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ECMooncakeConnectorMetadata):
            logger.error("Invalid metadata type for EC Mooncake Connector")
            return
            
        encoder_cache = kwargs.get('encoder_cache')
        mm_hash = kwargs.get('mm_hash')
        req_id = kwargs.get('request_id')
        input_id = kwargs.get('input_id')

        if not metadata.save_requests:
            return
            
        assert self.connector_worker is not None
        self.connector_worker.save_caches(metadata, encoder_cache, mm_hash, req_id=req_id, input_id=input_id)

    def wait_for_save(self):
        """Block until all save operations are complete"""
        if self.role != ECConnectorRole.WORKER:
            return
            
        assert self.connector_worker is not None
        self.connector_worker.wait_for_save()
        
    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get finished async operations"""
        if self.role != ECConnectorRole.WORKER:
            return None, None
            
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    # ==============================
    # Scheduler-side methods  
    # ==============================
    
    def check_caches_exist(self, request) -> list[bool]:
        """Check if encoder caches exist for each mm data of request"""
        if self.role != ECConnectorRole.SCHEDULER:
            return []
            
        assert self.connector_scheduler is not None
        return self.connector_scheduler.check_caches_exist(request)
        
    def update_state_after_alloc(self, request, index: int):
        """Update connector state after cache allocation"""
        if self.role != ECConnectorRole.SCHEDULER:
            return
            
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)
        
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECConnectorMetadata:
        """Build metadata for communication between scheduler and worker"""
        if self.role != ECConnectorRole.SCHEDULER:
            return ECMooncakeConnectorMetadata()
            
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)
        
    def request_finished(self, request, ec_connector_output: ECConnectorOutput) -> Optional[dict[str, Any]]:
        """Handle request completion"""
        if self.role != ECConnectorRole.SCHEDULER:
            return None

        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, ec_connector_output)


class ECMooncakeConnectorScheduler:
    """Scheduler-side implementation for EC Mooncake Connector"""
    
    def __init__(self, vllm_config: "VllmConfig", engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        
        logger.info(f"Initialized ECMooncakeConnectorScheduler for engine {engine_id}")
        
    def check_caches_exist(self, request) -> list[bool]:
        """Check cache existence for each mm data in request"""
        results = []
        
        # Extract mm_hashes from request
        mm_hashes = self._extract_mm_hashes(request)
        
        for mm_hash in mm_hashes:
            cache_key = self._generate_cache_key(mm_hash)
            exists = cache_key in self.cache_registry
            results.append(exists)
            
        return results
        
    def update_state_after_alloc(self, request, index: int):
        """Update state after allocation decision"""
        params = request.ec_transfer_params
        if params is not None:
            if all(p in params for p in ["ec_buffer_offset", "remote_engine_id", "remote_host", "remote_port"]):
                if request.request_id not in self._reqs_need_recv:
                    self._reqs_need_recv[request.request_id] = request
                    if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
                        self._ec_offsets_need_recv[request.request_id] = params["ec_buffer_offset"][request.request_id]
                    else:
                        mm_hash = request.mm_hashes[index]
                        if mm_hash not in self._ec_offsets_need_recv:
                            self._ec_offsets_need_recv[mm_hash] = params["ec_buffer_offset"][mm_hash]
            else:
                logger.warning(
                    "Incomplete ec_transfer_params for request %s: %s",
                    request.request_id, params)
        
        
                
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECMooncakeConnectorMetadata:
        """Build connector metadata for this scheduling step"""
        metadata = ECMooncakeConnectorMetadata(ec_offsets=self._ec_offsets_need_recv)
        
        for req_id, req in self._reqs_need_recv.items():
            metadata.add_new_request(
                request_id=req_id,
                mm_hashes=req.mm_hashes,
                ec_transfer_params=req.ec_transfer_params,
            )

        return metadata
        
    def request_finished(
        self,
        request: "Request",
        ec_connector_output: ECConnectorOutput,
    ) -> Optional[dict[str, Any]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.ec_transfer_params
        logger.debug(
            "ECMooncakeConnector request_finished, request_status=%s, "
            "ec_transfer_params=%s", request.status, params)

        if (params is None
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # TODO: Make ec_offset more precise,
        # redundant with other requests in batch now.
        return dict(
            ec_buffer_offset=ec_connector_output.ec_offset,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
        )
        
    def _extract_mm_hashes(self, request) -> List[str]:
        """Extract multimodal hashes from request"""
        # This needs to be implemented based on your request structure
        # Placeholder implementation
        if hasattr(request, 'mm_data') and request.mm_data:
            return [self._hash_mm_data(data) for data in request.mm_data]
        elif hasattr(request, 'mm_hashes') and request.mm_hashes:
            return request.mm_hashes
        return []
        
    def _hash_mm_data(self, mm_data: Any) -> str:
        """Generate hash for multimodal data"""
        # Simple hash implementation - enhance based on your needs
        return hashlib.md5(str(mm_data).encode()).hexdigest()
        
    def _generate_cache_key(self, mm_hash: str) -> str:
        """Generate cache key from mm_hash"""
        return f"ec_{self.engine_id}_{mm_hash}"
        
    def _cache_key_to_mm_hash(self, cache_key: str) -> str:
        """Extract mm_hash from cache_key"""
        return cache_key.split("_", 2)[-1]
        
    def _get_remote_info(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get remote engine info for loading cache"""
        # This should be implemented based on your distributed setup
        # Placeholder implementation
        return None


class ECMooncakeConnectorWorker:
    """Worker-side implementation for EC Mooncake Connector"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self._get_encoder_decoder_size(vllm_config)
        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)
        self.engine = TransferEngine()

        # Metadata.
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.encoder_cache: torch.Tensor = None
        self.side_channel_host = get_ip()
        self.max_device_id = self.tp_size * self.dp_size
        self.ec_role = vllm_config.ec_transfer_config.ec_role

        # Handshake base port
        self.side_channel_port = (
            vllm_config.ec_transfer_config.ec_port +
            vllm_config.parallel_config.data_parallel_rank_local *
            vllm_config.parallel_config.tensor_parallel_size)
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}

        # get tp device id
        # TODO(kw): https://github.com/vllm-project/vllm-ascend/pull/940
        # introducing some changes
        device_ids_str = envs_ascend.PHYSICAL_DEVICES
        if device_ids_str is None:
            device_ids = list(
                range(self.dp_rank * self.tp_size,
                      (self.dp_rank + 1) * self.tp_size))
        else:
            device_ids = list(map(int, device_ids_str.split(',')))
            start_index = self.dp_rank * self.tp_size
            end_index = start_index + self.tp_size
            if len(device_ids) < end_index:
                raise ValueError(
                    f"Not enough physical devices available for DP rank {self.dp_rank}. "
                    f"Expected at least {end_index} devices, but found {len(device_ids)} "
                    "in PHYSICAL_DEVICES.")
            device_ids = device_ids[start_index:end_index]
        assert len(device_ids) > self.tp_rank  # type: ignore
        self.device_id = device_ids[self.tp_rank]  # type: ignore

        self._initialize(
            hostname=self.side_channel_host + ':' + '0' + ':' + 'npu_' \
                     + str(self.device_id),
            device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()
        self.ec_role = vllm_config.ec_transfer_config.ec_role

        # Background thread for sending or receiving encoder caches.
        self.ec_send_thread: Optional[EncoderCacheSendingThread] = None
        self.ec_recv_thread: Optional[EncoderCacheRecvingThread] = None

        logger.info(f"Initialized ECMooncakeConnectorWorker for engine {engine_id}")
        
    def _initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        device_name = device_name if device_name is not None else ""
        ret_value = self.engine.initialize(hostname, "P2PHANDSHAKE", "ascend",
                                           device_name)
        if ret_value != 0:
            raise RuntimeError(
                f"Mooncake initialization failed with ret_value: {ret_value}")

    def register_cache(self, encoder_cache: torch.Tensor):
        """Register the Encoder Cache data."""

        base_addr = encoder_cache.data_ptr()
        region_len = encoder_cache.numel() * encoder_cache.element_size()
        self._register(base_addr, region_len)

        # After KV Caches registered, start the sending or receiving thread.
        metadata = ECMooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            encoder_cache_base_addr=base_addr,
        )

        ready_event = threading.Event()
        if self.ec_role == 'ec_producer':
            self.ec_send_thread = EncoderCacheSendingThread(self.tp_rank,
                                                       self._decoder_tp_size,
                                                       self.engine_id,
                                                       self.side_channel_host,
                                                       self.side_channel_port,
                                                       metadata, ready_event)
            self.ec_send_thread.start()
        else:
            self.ec_recv_thread = EncoderCacheRecvingThread(
                self.tp_rank, self.tp_size, self.engine, self.engine_id,
                self.handshake_port, base_addr, ready_event)
            self.ec_recv_thread.start()
        ready_event.wait()

    def _register(self, ptr, length):
        logger.info(
            "Registering encoder cache: ptr=0x%x, length=%d, ", ptr, length)
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            raise RuntimeError("Mooncake memory registration failed.")

    def _get_encoder_decoder_size(self, vllm_config: VllmConfig):
        # get encoder tp and dp size from extra config
        encoder_parallel_config: dict[
            str, Any] = vllm_config.ec_transfer_config.get_from_extra_config(
                "encoder", {})

        assert "tp_size" in encoder_parallel_config.keys()
        self._encoder_tp_size = encoder_parallel_config["tp_size"]

        assert "dp_size" in encoder_parallel_config.keys()
        self._encoder_dp_size = encoder_parallel_config["dp_size"]

        # get decoder tp and dp size from extra config
        decoder_parallel_config: dict[
            str, Any] = vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder", {})
        assert "tp_size" in decoder_parallel_config.keys()
        self._decoder_tp_size = decoder_parallel_config["tp_size"]
        assert "dp_size" in decoder_parallel_config.keys()
        self._decoder_dp_size = decoder_parallel_config["dp_size"]

    def save_caches(self, metadata: ECMooncakeConnectorMetadata,
                    encoder_cache: torch.Tensor, mm_hash: str,
                    req_id: Optional[str] = None,
                    input_id: Optional[str] = None):
        """Save encoder caches."""
        pass

    def start_load_caches(self, metadata: ECMooncakeConnectorMetadata,
                          encoder_cache: torch.Tensor):
        """Start loading encoder caches from remote engine."""
        assert self.ec_role == 'ec_consumer'
        assert self.ec_recv_thread is not None

        for req_id, meta in metadata.requests.items():
            self.ec_recv_thread.add_request(
                request_id=req_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_engine_id=meta.remote_engine_id,
                remote_host=meta.remote_host,
                remote_handshake_port=meta.remote_port,
            )
        if self.ec_send_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._get_remote_tp_ranks_for_req(req_id):
                    self.ec_send_thread.add_delayed_request(
                        req_id, delay_start_time)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.ec_send_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.ec_role == 'ec_producer' else set())
        done_recving = (
            self.ec_recv_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.ec_role == 'ec_consumer' else set())
        if self.tp_rank == 0:
            logger.debug(
                "Number of completed encoder cache send requests: %d, receive "
                "requests: %d", len(done_sending), len(done_recving))
        return done_sending, done_recving

    def start_load_ec(self, metadata: ECMooncakeConnectorMetadata):
        """Start loading encoder cache from remote engine."""
        for req_id, meta in metadata.requests.items():
            logger.debug(
                "start_load_ec for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                meta.remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))

            remote_handshake_port = meta.remote_port + \
                                    self._get_remote_tp_rank(req_id)
            self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                request_id=req_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_engine_id=meta.remote_engine_id,
                remote_host=meta.remote_host,
                remote_handshake_port=remote_handshake_port,
            )

        if self.kv_send_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._get_remote_tp_ranks_for_req(req_id):
                    self.kv_send_thread.add_delayed_request(
                        req_id, delay_start_time)

    def _get_remote_tp_rank(self, req_id: str) -> int:
        return self._get_remote_tp_ranks_for_req(req_id)[self.tp_rank]

    def _get_remote_tp_ranks_for_req(self, req_id: str) -> list[int]:
        if self._prefill_tp_size == self._decode_tp_size:
            return list(range(self._prefill_tp_size))

        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        sampled_nums = rand.sample(range(self._prefill_tp_size),
                                   self._decode_tp_size)
        return sampled_nums


# Export list for public API
__all__ = [
    "ECMooncakeConnector",
    "ECMooncakeConnectorMetadata", 
    "ECRequestMetadata",
    "ECMooncakeConnectorScheduler",
    "ECMooncakeConnectorWorker"
]