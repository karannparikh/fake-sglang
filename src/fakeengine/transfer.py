"""The simulated KV-cache transfer.

Walks SGLang's ``KVPoll`` lifecycle honestly -- ``Bootstrapping`` is a real
HTTP wait on the peer, ``Transferring`` is a sleep sized by the actual payload
(``tokens x kv_bytes_per_token / bandwidth``) -- and only the bytes are
imaginary. Keeping the states real is what makes the timing observable: raise
``--fake-kv-bandwidth-gb-s`` and measured TTFT drops, which is the evidence
that the transfer is modelled rather than skipped.

``FAKE_BOOTSTRAP_HOST`` short-circuits the whole thing, matching SGLang's
``_is_fake_transfer``. Dynamo's health-check canary and
``_disagg.warmup_prefill_engine`` both send that host and would otherwise wait
on a peer that is never coming.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, Optional

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, KVPoll

from fakeengine.bootstrap import BootstrapClient
from fakeengine.config import SimConfig
from fakeengine.timing import TimingModel
from fakeengine.topology import Link

logger = logging.getLogger(__name__)


class KVReceiver:
    """Decode-side receiver for one request."""

    def __init__(
        self,
        config: SimConfig,
        client: BootstrapClient,
        *,
        bootstrap_host: str,
        bootstrap_port: int,
        bootstrap_room: int,
    ) -> None:
        self.config = config
        self.client = client
        self.host = bootstrap_host
        self.port = int(bootstrap_port)
        self.room = int(bootstrap_room)
        self.state = KVPoll.Bootstrapping
        self.metadata: Dict[str, Any] = {}
        self.error: Optional[BaseException] = None
        self._timing = TimingModel(config)

    @property
    def is_fake_host(self) -> bool:
        return self.host == FAKE_BOOTSTRAP_HOST

    def poll(self) -> int:
        return self.state

    async def receive(self, prompt_len: int) -> Dict[str, Any]:
        """Run Bootstrapping -> WaitingForInput -> Transferring -> Success."""
        if self.is_fake_host:
            # SGLang's sentinel: no peer, no transfer, no wait.
            self.state = KVPoll.Success
            self.metadata = {"num_tokens": prompt_len, "kv_bytes": 0, "fake": True}
            return self.metadata

        try:
            self.metadata = await self.client.await_room(self.host, self.port, self.room)
            self.state = KVPoll.WaitingForInput
        except BaseException as exc:
            self.state = KVPoll.Failed
            self.error = exc
            raise

        num_tokens = int(self.metadata.get("num_tokens") or prompt_len)
        self.state = KVPoll.Transferring
        link = self._link(self.metadata.get("location"))
        transfer_ms = self._timing.transfer_ms(num_tokens, link)
        mib = num_tokens * self.config.kv_bytes_per_token / 1024 / 1024
        if link is not None:
            # INFO, not DEBUG: which link each handoff crossed is the whole
            # point of running a topology, and grepping for it is how a run is
            # judged.
            logger.info(
                "fake-sglang: KV room %d %s -> %s over %r link (%g GB/s): "
                "%d tokens, %.2f MiB, %.2f ms",
                self.room,
                self.metadata.get("location"),
                self.config.location,
                link.name,
                link.bandwidth_gb_s,
                num_tokens,
                mib,
                transfer_ms,
            )
        else:
            logger.debug(
                "fake-sglang: room %d transferring %d tokens (%.2f MiB) in %.2f ms",
                self.room,
                num_tokens,
                mib,
                transfer_ms,
            )
        await self._timing.sleep_ms(transfer_ms)

        self.state = KVPoll.Success
        self.metadata["transfer_ms"] = transfer_ms
        self.metadata["link"] = link.name if link else None
        await self.client.release_room(self.host, self.port, self.room)
        return self.metadata

    def abort(self) -> None:
        self.state = KVPoll.Failed

    def _link(self, peer_location: Optional[str]) -> Optional[Link]:
        """The link to the prefill peer, or ``None`` for the global one.

        Needs a topology plus a location at both ends; a worker launched
        without placement keeps the old single-bandwidth behaviour.
        """
        topology = self.config.topology
        if topology is None or not peer_location or not self.config.location:
            return None
        return topology.link_between(peer_location, self.config.location)


class KVSender:
    """Prefill-side sender for one request. Publishing the room *is* the send:
    the decode peer's poll is what observes it."""

    def __init__(
        self,
        config: SimConfig,
        server,
        bootstrap_room: int,
        *,
        dp_rank: int,
    ) -> None:
        self.config = config
        self.server = server
        self.room = int(bootstrap_room)
        self.dp_rank = int(dp_rank)
        self.state = KVPoll.Bootstrapping

    def poll(self) -> int:
        return self.state

    def send(self, num_tokens: int, first_token: Optional[int] = None) -> None:
        if self.server is None:
            self.state = KVPoll.Failed
            return
        self.state = KVPoll.Transferring
        self.server.publish_room(
            self.room,
            num_tokens=num_tokens,
            kv_bytes=num_tokens * self.config.kv_bytes_per_token,
            first_token=first_token,
            location=self.config.location,
            kv_return=self._return_ticket(),
        )
        self.state = KVPoll.Success

    def fail(self, reason: str) -> None:
        self.state = KVPoll.Failed
        if self.server is not None:
            self.server.fail_room(self.room, reason)

    def _return_ticket(self) -> Optional[Dict[str, Any]]:
        if not (
            self.config.enable_kv_return
            and self.config.disaggregation_mode == "prefill"
            and self.config.prefix_caching
        ):
            return None
        return {
            "version": self.config.kv_return_protocol_version,
            "enabled": True,
            "nonce": secrets.token_urlsafe(24),
            "prefill_dp_rank": self.dp_rank,
            "max_return_tokens": self.config.kv_return_max_tokens,
            "deadline_ms": self.config.kv_return_deadline_ms,
        }
