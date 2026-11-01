"""The prefill/decode rendezvous.

This is a *real* cross-process handshake over HTTP, not a mocked one. That
matters: it is the part of disaggregation most likely to be misconfigured in a
deployment (wrong advertised host, port collision, dead peer), and a fake that
skipped it would hide exactly the failures worth rehearsing. Only the KV
payload itself is simulated -- see :mod:`fakeengine.transfer`.

The route shape follows SGLang's ``CommonKVBootstrapServer``
(``disaggregation/common/conn.py``): ``GET /health``, ``PUT /route`` for rank
registration, ``GET /route`` for topology. ``/room/{id}`` is ours -- SGLang
carries per-request metadata over ZMQ rather than HTTP, and HTTP keeps the fake
engine dependency-light.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class BootstrapServer:
    """Runs in the prefill worker. Publishes rooms as their KV becomes ready."""

    def __init__(self, host: str, port: int, *, room_ttl_s: float = 120.0) -> None:
        self.host = host
        self.port = port
        self.room_ttl_s = room_ttl_s
        self._rooms: Dict[int, Dict[str, Any]] = {}
        self._ranks: Dict[str, Dict[str, Any]] = {}
        self._runner: Optional[web.AppRunner] = None
        self._sweeper: Optional[asyncio.Task] = None

    async def start(self) -> None:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/route", self._get_route),
                web.put("/route", self._put_route),
                web.get("/room/{room}", self._get_room),
                web.delete("/room/{room}", self._delete_room),
            ]
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        # Bind every interface but advertise a specific host: a decode worker
        # may reach us over loopback or the LAN address, and which one it uses
        # is decided by whatever Dynamo published in the model card.
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        self._sweeper = asyncio.get_running_loop().create_task(self._sweep_rooms())
        logger.info(
            "fake-sglang bootstrap server listening on 0.0.0.0:%d (advertising %s)",
            self.port,
            self.host,
        )

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _sweep_rooms(self) -> None:
        """Drop rooms whose decode peer never came.

        A room is normally released by the receiver once the transfer lands. If
        the decode worker dies, is never scheduled, or the request is cancelled
        between the two legs, nothing releases it -- and a long-lived prefill
        worker would accumulate them forever. SGLang sweeps the same way, on
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT.
        """
        interval = max(self.room_ttl_s / 4, 1.0)
        while True:
            await asyncio.sleep(interval)
            cutoff = time.time() - self.room_ttl_s
            stale = [
                room
                for room, entry in self._rooms.items()
                if entry.get("ready_ts", 0) < cutoff
            ]
            for room in stale:
                self._rooms.pop(room, None)
            if stale:
                logger.warning(
                    "fake-sglang: expired %d bootstrap room(s) never claimed by a "
                    "decode peer (oldest > %.0fs): %s",
                    len(stale),
                    self.room_ttl_s,
                    stale[:8],
                )

    def publish_room(
        self,
        room: int,
        *,
        num_tokens: int,
        kv_bytes: int,
        first_token: Optional[int] = None,
        location: Optional[str] = None,
        kv_return: dict[str, Any] | None = None,
    ) -> None:
        # ``first_token`` is prefill's real output. SGLang hands the same value
        # to decode through its metadata buffer rather than the KV payload
        # (``disaggregation/decode.py::_commit_transfer_to_req``); carrying it
        # here is what makes a disaggregated answer identical to an aggregated
        # one for the same prompt.
        metadata: dict[str, Any] = {
            "bootstrap_room": int(room),
            "num_tokens": int(num_tokens),
            "kv_bytes": int(kv_bytes),
            "first_token": first_token,
            "location": location,
            "ready_ts": time.time(),
        }
        if kv_return is not None:
            metadata["kv_return"] = kv_return
        self._rooms[int(room)] = metadata

    def fail_room(self, room: int, reason: str) -> None:
        self._rooms[int(room)] = {
            "bootstrap_room": int(room),
            "error": reason,
            "ready_ts": time.time(),
        }

    # -- handlers ----------------------------------------------------------

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "engine": "fake-sglang", "rooms": len(self._rooms)}
        )

    async def _get_route(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "engine": "fake-sglang",
                "prefill_host": self.host,
                "prefill_port": self.port,
                "ranks": self._ranks,
                "follow_bootstrap_room": True,
            }
        )

    async def _put_route(self, request: web.Request) -> web.Response:
        payload = await request.json()
        key = str(payload.get("rank_id", len(self._ranks)))
        self._ranks[key] = payload
        return web.json_response({"status": "registered", "rank_id": key})

    async def _get_room(self, request: web.Request) -> web.Response:
        room = int(request.match_info["room"])
        entry = self._rooms.get(room)
        if entry is None:
            # 404 is the "still bootstrapping" signal, not an error: the decode
            # worker can legitimately arrive before prefill has finished.
            return web.json_response({"status": "pending", "bootstrap_room": room}, status=404)
        if "error" in entry:
            return web.json_response(entry, status=410)
        return web.json_response(entry)

    async def _delete_room(self, request: web.Request) -> web.Response:
        room = int(request.match_info["room"])
        self._rooms.pop(room, None)
        return web.json_response({"status": "released", "bootstrap_room": room})


class BootstrapClient:
    """Runs in the decode worker. Polls a prefill peer for one room."""

    def __init__(self, poll_interval_s: float, timeout_s: float) -> None:
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._session: Any = None

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s + 5)
            )
        return self._session

    async def await_room(self, host: str, port: int, room: int) -> Dict[str, Any]:
        """Block until the prefill peer publishes ``room``.

        Raises ``TimeoutError`` rather than hanging, so a dead or misaddressed
        prefill worker surfaces as a failed request instead of a stuck one.
        """
        import aiohttp

        session = await self._get_session()
        url = f"http://{_bracket(host)}:{port}/room/{room}"
        deadline = time.monotonic() + self.timeout_s

        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 410:
                        payload = await response.json()
                        raise RuntimeError(
                            f"prefill reported failure for room {room}: "
                            f"{payload.get('error')}"
                        )
            except aiohttp.ClientError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not reach prefill bootstrap at {url}: {exc}"
                    ) from exc

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {self.timeout_s:.0f}s waiting for KV of "
                    f"bootstrap_room {room} from {host}:{port}"
                )
            await asyncio.sleep(self.poll_interval_s)

    async def release_room(self, host: str, port: int, room: int) -> None:
        try:
            session = await self._get_session()
            url = f"http://{_bracket(host)}:{port}/room/{room}"
            async with session.delete(url):
                pass
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail a request
            logger.debug("fake-sglang: room %s release failed: %s", room, exc)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def _bracket(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host
