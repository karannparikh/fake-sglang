"""``sglang.Engine`` -- the translation layer over :mod:`fakeengine`.

Everything SGLang-shaped lives here; everything behavioral lives in
``fakeengine``. The contract this must satisfy is set by Dynamo's SGLang
backend, chiefly:

* ``prefill_handler.py`` -- ``async_generate(..., bootstrap_host/port/room)``
  and a stream it can drain in the background after yielding the handshake.
* ``decode_handler.py::_process_token_stream`` -- chunks of
  ``{"output_ids": [...], "text": str, "meta_info": {...}}`` where
  ``output_ids`` is disjoint per chunk.
* ``handler_base.py`` -- ``engine.tokenizer_manager`` for aborts, cache
  flushes, profiling and weight-update routes.
* ``register.py`` -- ``engine._scheduler_init_result.scheduler_infos[0]`` for
  the KV capacity it publishes in the model card.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, ExitStack, aclosing
from itertools import count
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

from fakeengine import tokens as token_gen
from fakeengine.bootstrap import BootstrapClient, BootstrapServer
from fakeengine.cache_metrics import PrefixCacheMetrics
from fakeengine.config import SimConfig, announce
from fakeengine.fpm import ForwardPassPublisher
from fakeengine.kv_events import build_emitter
from fakeengine.scheduler import Request, Scheduler
from fakeengine.stats import MetricsPusher
from fakeengine.transfer import KVReceiver, KVSender

logger = logging.getLogger(__name__)


class _SchedulerInitResult:
    """Stands in for SGLang's scheduler handshake payload.

    ``register.py`` reads ``scheduler_infos[0]`` to publish ``total_kv_blocks``
    and ``max_num_batched_tokens`` on the model card; without it the router has
    no capacity figure for this worker.
    """

    def __init__(self, config: SimConfig, dp_size: int = 1) -> None:
        self.scheduler_infos: List[Dict[str, Any]] = [
            {
                "max_total_num_tokens": config.total_kv_tokens,
                "max_req_input_len": config.context_len,
                "max_prefill_tokens": config.max_batched_tokens,
                "max_running_requests": config.max_running_requests,
                "hicache_host_total_tokens": 0,
            } for _ in range(dp_size)
        ]


class TokenizerManager:
    """The control-plane surface Dynamo reaches through ``engine.tokenizer_manager``."""

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine
        self.server_args = engine.server_args
        self.model_config = engine.model_config
        self.context_len = engine.sim_config.context_len
        self.is_generation = not engine.server_args.is_embedding
        self.tokenizer = None            # Dynamo tokenizes; see --skip-tokenizer-init
        self.rid_to_state: Dict[str, Any] = {}
        self.initial_weights_loaded = True
        self.is_pause = False
        self.updated_version: Optional[str] = None
        self.num_reserved_tokens = 0

    # -- request path ------------------------------------------------------

    async def generate_request(self, obj: Any, request: Any = None) -> AsyncIterator[dict]:
        """SGLang-native ``/generate``. Dynamo routes the ``sglang_tito``
        passthrough here rather than through ``async_generate``."""
        stream = await self._engine.async_generate(
            input_ids=getattr(obj, "input_ids", None),
            sampling_params=getattr(obj, "sampling_params", None) or {},
            stream=True,
            rid=getattr(obj, "rid", None),
            bootstrap_host=getattr(obj, "bootstrap_host", None),
            bootstrap_port=getattr(obj, "bootstrap_port", None),
            bootstrap_room=getattr(obj, "bootstrap_room", None),
            return_logprob=getattr(obj, "return_logprob", False),
            top_logprobs_num=getattr(obj, "top_logprobs_num", 0),
            cache_salt=getattr(obj, "cache_salt", None),
            routed_dp_rank=getattr(obj, "routed_dp_rank", None),
            data_parallel_rank=getattr(obj, "data_parallel_rank", None),
        )
        async with aclosing(stream):
            async for chunk in stream:
                yield chunk

    def abort_request(self, rid: str = "", abort_all: bool = False) -> None:
        for scheduler in self._engine.schedulers:
            if abort_all:
                for pending in list(scheduler.by_rid):
                    scheduler.abort(pending)
            elif rid:
                scheduler.abort(rid)

    # -- cache / memory ----------------------------------------------------

    async def flush_cache(self) -> Any:
        for scheduler in self._engine.schedulers:
            scheduler.flush_cache()
        return type("FlushResult", (), {"success": True})()

    async def clear_hicache_storage(self) -> Any:
        return type("ClearResult", (), {"success": True})()

    async def release_memory_occupation(self, obj: Any = None) -> None:
        return None

    async def resume_memory_occupation(self, obj: Any = None) -> None:
        return None

    async def pause_generation(self, obj: Any = None) -> None:
        self.is_pause = True

    async def continue_generation(self, obj: Any = None) -> None:
        self.is_pause = False

    def auto_create_handle_loop(self) -> None:
        self._engine.start_background()

    def validate_total_tokens(self, *args: Any, **kwargs: Any) -> bool:
        return True

    # -- profiling ---------------------------------------------------------

    async def start_profile(self, obj: Any = None) -> None:
        logger.info("fake-sglang: profiling is a no-op (no forward pass to profile)")

    async def stop_profile(self, obj: Any = None) -> None:
        return None

    # -- weights / LoRA / elastic EP --------------------------------------

    def _unsupported(self, what: str):
        raise NotImplementedError(f"fake-sglang has no weights, so {what} is unsupported")

    async def update_weights_from_disk(self, obj: Any = None):
        self._unsupported("update_weights_from_disk")

    async def update_weights_from_distributed(self, obj: Any = None):
        self._unsupported("update_weights_from_distributed")

    async def update_weights_from_tensor(self, obj: Any = None):
        self._unsupported("update_weights_from_tensor")

    async def update_weights_from_ipc(self, obj: Any = None):
        self._unsupported("update_weights_from_ipc")

    def _update_weight_version_if_provided(self, version: Optional[str]) -> None:
        if version:
            self.updated_version = version

    async def load_lora_adapter(self, obj: Any = None):
        self._unsupported("load_lora_adapter")

    async def unload_lora_adapter(self, obj: Any = None):
        self._unsupported("unload_lora_adapter")

    async def scale_elastic_ep(self, obj: Any = None):
        self._unsupported("scale_elastic_ep")

    async def get_elastic_ep_state(self) -> Dict[str, Any]:
        return {"ep_size": int(getattr(self.server_args, "ep_size", 1) or 1)}


class Engine:
    """CPU simulation of an SGLang engine."""

    # Override point Dynamo's NIXL telemetry wrapper patches. There is no
    # scheduler subprocess here, so it is never invoked -- but its absence
    # raises in `install_per_rank_nixl_prometheus_ports`.
    run_scheduler_process_func = None

    def __init__(self, server_args: Optional[ServerArgs] = None, **kwargs: Any) -> None:
        if server_args is None:
            server_args = ServerArgs(**kwargs)
        self.server_args = server_args
        self.model_config = server_args.get_model_config()
        self.sim_config = SimConfig.from_server_args(server_args)
        if server_args.dp_size > 1 and (
            not server_args.enable_dp_attention or server_args.nnodes != 1
            or server_args.node_rank != 0 or server_args.is_embedding
        ):
            raise ValueError(
                "Packed fake workers require --enable-dp-attention, one node, "
                "node rank zero, and a generation model"
            )
        self.sim_config.bootstrap_host = _advertised_host(server_args)
        announce(self.sim_config)

        # Publishing starts with the engine, not with the first request: the
        # router subscribes as soon as the worker registers, and a socket that
        # appears late drops the events published before it.
        self.schedulers: List[Scheduler] = []
        self._rank_events = []
        self._rank_metrics: List[MetricsPusher] = []
        self._unrouted_rank = count()
        self.port_args = PortArgs(server_args)
        with ExitStack() as cleanup:
            for rank in range(server_args.dp_size):
                events = build_emitter(
                    server_args, self.sim_config.block_size, dp_rank=rank,
                )
                cleanup.callback(events.shutdown)
                self._rank_events.append(events)
                cache_metrics = (
                    PrefixCacheMetrics(dp_rank=rank) if server_args.enable_metrics else None
                )
                fpm = None
                if server_args.enable_forward_pass_metrics:
                    fpm = ForwardPassPublisher(
                        server_args.forward_pass_metrics_ipc_name,
                        server_args.forward_pass_metrics_worker_id,
                        dp_rank=rank,
                    )
                    cleanup.callback(fpm.close)
                scheduler = Scheduler(
                    self.sim_config, kv_events=events,
                    fpm=fpm, cache_metrics=cache_metrics,
                )
                self.schedulers.append(scheduler)
                self._rank_metrics.append(MetricsPusher(
                    scheduler, self.port_args.metrics_ipc_name, dp_rank=rank,
                ))
            # Preserve the single-rank inspection surface used by existing tools.
            self.scheduler = self.schedulers[0]
            self.kv_events = self._rank_events[0]
            self._metrics = self._rank_metrics[0]
            self._scheduler_init_result = _SchedulerInitResult(
                self.sim_config, server_args.dp_size,
            )
            self.tokenizer_manager = TokenizerManager(self)

            self._bootstrap_server: Optional[BootstrapServer] = None
            self._bootstrap_client: Optional[BootstrapClient] = None
            if self.disaggregation_mode == "prefill":
                self._bootstrap_server = BootstrapServer(
                    self.sim_config.bootstrap_host,
                    self.sim_config.bootstrap_port,
                    room_ttl_s=max(self.sim_config.bootstrap_timeout_s * 4, 60.0),
                )
            elif self.disaggregation_mode == "decode":
                self._bootstrap_client = BootstrapClient(
                    self.sim_config.bootstrap_poll_interval_s,
                    self.sim_config.bootstrap_timeout_s,
                )
            cleanup.pop_all()
        if server_args.dp_size > 1:
            logger.info(
                "fake-sglang: packed %d independent DP ranks in one process; "
                "%d KV tokens and %d running slots per rank",
                server_args.dp_size, self.sim_config.total_kv_tokens,
                self.sim_config.max_running_requests,
            )
        self._bootstrap_ready: Optional[asyncio.Task] = None
        self._started = False
        self._shutdown = False

        self.start_background()

    @property
    def disaggregation_mode(self) -> str:
        return self.sim_config.disaggregation_mode

    # -- lifecycle ---------------------------------------------------------

    def start_background(self) -> None:
        """Start the scheduler and metrics push without awaiting.

        Load must be reported from worker startup, not from the first request:
        a worker that reports nothing until it is already busy is invisible to
        load-based routing exactly when routing matters.
        """
        if self._shutdown:
            raise RuntimeError("Engine has been shut down")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        for scheduler, metrics in zip(self.schedulers, self._rank_metrics):
            scheduler.start()
            metrics.start()
        if self._bootstrap_server is not None and self._bootstrap_ready is None:
            self._bootstrap_ready = asyncio.get_running_loop().create_task(
                self._bootstrap_server.start()
            )

    async def _ensure_started(self) -> None:
        self._started = True
        self.start_background()
        if self._bootstrap_ready is not None:
            # Bind before the first request is answered: a decode peer can
            # reach this port as soon as Dynamo publishes the disaggregated
            # endpoint in the model card.
            await self._bootstrap_ready

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called outside the loop (atexit, a sync teardown): run it now,
            # otherwise the bootstrap server and HTTP session leak.
            asyncio.run(self._async_shutdown())
            return
        loop.create_task(self._async_shutdown())

    async def _async_shutdown(self) -> None:
        self._shutdown = True
        # Unwind every resource even if an earlier task failed.
        async with AsyncExitStack() as cleanup:
            if self._bootstrap_server is not None:
                cleanup.push_async_callback(self._bootstrap_server.stop)
            if self._bootstrap_client is not None:
                cleanup.push_async_callback(self._bootstrap_client.close)
            for events in self._rank_events:
                cleanup.callback(events.shutdown)
            for scheduler in self.schedulers:
                cleanup.push_async_callback(scheduler.stop)
            for metrics in self._rank_metrics:
                cleanup.push_async_callback(metrics.stop)
            if self._bootstrap_ready is not None:
                self._bootstrap_ready.cancel()
                try:
                    await self._bootstrap_ready
                except asyncio.CancelledError:
                    pass

    # -- generation --------------------------------------------------------

    async def async_generate(
        self,
        *,
        input_ids: Optional[List[int]] = None,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        rid: Optional[str] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        cache_salt: Optional[Union[List[str], str]] = None,
        routed_dp_rank: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        """Returns the stream; the caller awaits this, then iterates it.

        Matches SGLang, and Dynamo depends on the shape: ``prefill_handler``
        does ``results = await engine.async_generate(...)`` and then hands
        ``results`` to a background drain task.
        """
        rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank,
        )
        if rank is None:
            rank = next(self._unrouted_rank) % len(self.schedulers)
        await self._ensure_started()

        prompt_ids = _resolve_prompt_ids(input_ids, prompt, self.sim_config.vocab_size)
        params = dict(sampling_params or {})
        max_new = int(params.get("max_new_tokens") or 128)
        ignore_eos = bool(params.get("ignore_eos", False))
        min_new = int(params.get("min_new_tokens") or 0)
        request_id = str(rid or uuid.uuid4().hex)

        return self._generate_stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            cache_salt=_single_salt(cache_salt),
            scheduler=self.schedulers[rank],
            ignore_eos=ignore_eos,
            min_new_tokens=min_new,
        )

    def _resolve_routed_dp_rank(
        self,
        routed_dp_rank: Optional[int],
        data_parallel_rank: Optional[int],
    ) -> Optional[int]:
        """SGLang's DP-rank validation, against this worker's ``dp_size``.

        Dynamo reads this off the class (``gateway.GatewayEngine``), so it must
        exist on the fake engine too. Same rules as SGLang:
        ``data_parallel_rank`` is the deprecated spelling, rank 0 on a single
        rank is a no-op, and anything out of range is an error rather than a
        silent reroute.
        """
        if data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if routed_dp_rank is None:
                routed_dp_rank = data_parallel_rank

        if routed_dp_rank is None:
            return None
        if isinstance(routed_dp_rank, bool) or not isinstance(routed_dp_rank, int):
            raise ValueError(f"Invalid routed DP rank {routed_dp_rank!r}: expected an integer")
        num_dp_ranks = self.server_args.dp_size
        if num_dp_ranks <= 1 and routed_dp_rank == 0:
            return None
        if routed_dp_rank < 0 or routed_dp_rank >= num_dp_ranks:
            raise ValueError(
                f"routed_dp_rank={routed_dp_rank} out of range [0, {num_dp_ranks})"
            )
        return routed_dp_rank

    async def async_encode(self, **kwargs: Any) -> Dict[str, Any]:
        await self._ensure_started()
        input_ids = kwargs.get("input_ids") or []
        dimension = min(self.model_config.hidden_size, 64)
        seed = token_gen.seed_for(input_ids)
        embedding = [((seed >> (i % 48)) % 2000) / 1000.0 - 1.0 for i in range(dimension)]
        return {
            "embedding": embedding,
            "meta_info": {"id": uuid.uuid4().hex, "prompt_tokens": len(input_ids)},
        }

    async def _generate_stream(
        self,
        *,
        request_id: str,
        prompt_ids: List[int],
        max_new_tokens: int,
        bootstrap_host: Optional[str],
        bootstrap_port: Optional[int],
        bootstrap_room: Optional[int],
        scheduler: Scheduler,
        cache_salt: Optional[str] = None,
        ignore_eos: bool = False,
        min_new_tokens: int = 0,
    ) -> AsyncIterator[dict]:
        mode = self.disaggregation_mode
        started = time.monotonic()

        if mode == "decode" and bootstrap_room is not None:
            async with aclosing(self._decode_stream(
                request_id,
                prompt_ids,
                max_new_tokens,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                started,
                scheduler,
                ignore_eos,
                min_new_tokens,
            )) as stream:
                async for chunk in stream:
                    yield chunk
            return

        request = Request(
            request_id, prompt_ids, max_new_tokens, cache_salt=cache_salt,
            # Dynamo's disaggregated canaries and warmup use a no-peer transfer.
            observe_cache=bootstrap_host != FAKE_BOOTSTRAP_HOST,
            ignore_eos=ignore_eos,
            min_new_tokens=min_new_tokens,
        )
        scheduler.submit(request)

        sender: Optional[KVSender] = None
        if mode == "prefill" and bootstrap_room is not None:
            sender = KVSender(self.sim_config, self._bootstrap_server, bootstrap_room)

        published = False
        try:
            async for chunk in self._drain(request, started):
                # Publish the room as soon as prefill has produced its token:
                # that is the moment this worker's KV for the prompt exists.
                # The token goes with it, the way SGLang hands its real first
                # output to decode through the metadata buffer.
                if sender is not None and not published:
                    published = True
                    if bootstrap_host != FAKE_BOOTSTRAP_HOST:
                        first = chunk["output_ids"][0] if chunk["output_ids"] else None
                        sender.send(len(prompt_ids), first_token=first)
                yield chunk
        except BaseException as exc:
            scheduler.abort(request_id)
            if sender is not None and not published:
                sender.fail(f"{type(exc).__name__}: {exc}")
            raise

    async def _decode_stream(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_new_tokens: int,
        bootstrap_host: Optional[str],
        bootstrap_port: Optional[int],
        bootstrap_room: int,
        started: float,
        scheduler: Scheduler,
        ignore_eos: bool = False,
        min_new_tokens: int = 0,
    ) -> AsyncIterator[dict]:
        """Wait for the peer's KV, then decode from a prebuilt batch."""
        receiver = KVReceiver(
            self.sim_config,
            self._bootstrap_client,
            bootstrap_host=bootstrap_host or FAKE_BOOTSTRAP_HOST,
            bootstrap_port=bootstrap_port or self.sim_config.bootstrap_port,
            bootstrap_room=bootstrap_room,
        )
        metadata = await receiver.receive(len(prompt_ids))

        # Prefill already emitted one token; decode continues from there, so
        # that token leads this stream. Prefer the one prefill actually
        # produced -- that is what makes a disaggregated answer byte-identical
        # to the aggregated answer for the same prompt, and reproducible across
        # runs. Only the no-peer path (FAKE_BOOTSTRAP_HOST, health checks and
        # warmup) has no real token to inherit, and falls back to SGLang's
        # request-id hash.
        handoff = metadata.get("first_token")
        if handoff is None:
            handoff = token_gen.handoff_token(request_id, self.sim_config.vocab_size)
        request = Request(
            request_id,
            prompt_ids,
            max(max_new_tokens, 1),
            prebuilt=True,
            first_token=handoff,
            ignore_eos=ignore_eos,
            min_new_tokens=min_new_tokens,
        )
        scheduler.submit(request)
        try:
            async for chunk in self._drain(request, started):
                yield chunk
        finally:
            scheduler.abort(request_id)

    async def _drain(self, request: Request, started: float) -> AsyncIterator[dict]:
        """Turn scheduler events into SGLang response chunks.

        One token is held back so the terminal chunk can carry both the last
        token and the finish reason, the way SGLang's streaming does.
        """
        completion = 0
        pending: Optional[int] = None

        while True:
            kind, payload = await request.queue.get()
            if kind == "token":
                if pending is not None:
                    completion += 1
                    yield self._chunk(request, [pending], completion, None, started)
                pending = payload
                continue

            # kind == "done"
            tokens: List[int] = []
            if pending is not None:
                tokens.append(pending)
                completion += 1
            yield self._chunk(request, tokens, completion, payload, started)
            return

    def _chunk(
        self,
        request: Request,
        output_ids: List[int],
        completion_tokens: int,
        finish_reason: Optional[str],
        started: float,
    ) -> dict:
        meta: Dict[str, Any] = {
            "id": request.rid,
            "prompt_tokens": len(request.prompt_ids),
            "completion_tokens": completion_tokens,
            "cached_tokens": request.cached_tokens,
            "e2e_latency": time.monotonic() - started,
            "finish_reason": None,
        }
        if finish_reason is not None:
            meta["finish_reason"] = {"type": finish_reason, "matched": None}
        return {"text": "", "output_ids": list(output_ids), "meta_info": meta}


def _single_salt(cache_salt: Optional[Union[List[str], str]]) -> Optional[str]:
    """SGLang's ``_normalize_cache_salt`` for the one prompt the fake serves:
    an empty salt is no salt, and a batch must give strings."""
    if isinstance(cache_salt, (list, tuple)):
        if any(not isinstance(value, str) for value in cache_salt):
            raise ValueError("Every cache_salt should be a string.")
        cache_salt = cache_salt[0] if cache_salt else None
    elif cache_salt is not None and not isinstance(cache_salt, str):
        raise ValueError("cache_salt should be a string.")
    return cache_salt or None


def _advertised_host(server_args: ServerArgs) -> str:
    """The host a decode peer should dial to reach this worker's bootstrap.

    Mirrors Dynamo's own resolution in ``_disagg.compute_bootstrap_address`` so
    the address this engine binds and the address Dynamo publishes agree.
    """
    if server_args.dist_init_addr:
        return NetworkAddress.parse(server_args.dist_init_addr).resolved().host
    return get_local_ip_auto()


def _resolve_prompt_ids(
    input_ids: Optional[List[int]], prompt: Optional[str], vocab_size: int
) -> List[int]:
    if input_ids:
        first = input_ids[0]
        # SGLang accepts a batch of prompts; the fake engine takes the first.
        return list(first) if isinstance(first, (list, tuple)) else list(input_ids)
    if prompt:
        # No tokenizer here: hash the text into stable pseudo-token ids so a
        # text-mode request still gets deterministic, length-proportional work.
        text = prompt if isinstance(prompt, str) else str(prompt)
        return [
            (token_gen.seed_for([ord(c)]) % max(vocab_size - 1, 1)) + 1
            for c in text[:4096]
        ] or [1]
    return [1]
