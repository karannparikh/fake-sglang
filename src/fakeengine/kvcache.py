"""KV block accounting and prefix reuse.

Real enough to produce the pressure the scheduler must react to, and -- since
the pool's contents are published as KV events -- real enough that the router's
view of what is resident on this worker stays truthful. A block leaves the
prefix cache exactly when it stops being usable, whether that is eviction under
pressure or reuse of a block that was merely unreferenced.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


def namespace_seed(cache_salt: Optional[str]) -> bytes:
    """Where a salted request's hash chain starts.

    SGLang keys its prefix cache by the request's ``cache_salt`` as well as its
    tokens, so tenants with different salts never share KV even for an
    identical prompt. Seeding the chain gives the same property: every block
    hash of a salted sequence depends on the salt. No salt keeps the empty
    seed, so unsalted ids are unchanged. The length prefix keeps "a"+"bc" and
    "ab"+"c" apart, as SGLang's ``kv_event_namespace_seed`` does.
    """
    if not cache_salt:
        return b""
    encoded = cache_salt.encode("utf-8")
    return hashlib.blake2b(
        b"fake-sglang-cache-salt-v1" + len(encoded).to_bytes(8, "little") + encoded,
        digest_size=16,
    ).digest()


def block_hashes(
    token_ids: Sequence[int], block_size: int, cache_salt: Optional[str] = None
) -> List[bytes]:
    """Cumulative (prefix-inclusive) hash per full block.

    Cumulative rather than per-block: two prompts sharing a suffix but not a
    prefix must NOT match, and only a chained hash expresses that. The chain
    starts from the salt's seed (``namespace_seed``).

    Trailing tokens that do not fill a block produce no hash. That is
    deliberate -- a partial block is not a cache entry, and Dynamo drops any
    published block whose token count is not exactly ``block_size``.
    """
    hashes: List[bytes] = []
    running = namespace_seed(cache_salt)
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        chunk = token_ids[start : start + block_size]
        running = hashlib.blake2b(
            running + b",".join(str(t).encode() for t in chunk),
            digest_size=16,
        ).digest()
        hashes.append(running)
    return hashes


def external_id(digest: bytes) -> int:
    """The engine-side block id published to the router.

    Dynamo treats this as opaque -- it keys its tree by it for removal but
    recomputes the prefix-matching hash from the token ids itself
    (``lib/kv-router/src/zmq_wire/convert.rs``). Deriving it from the content
    hash means identical prefixes get identical ids, which is what a radix
    cache does and what lets the router dedupe across requests.

    Masked to 63 bits so it survives msgpack's signed integer encoding.
    """
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class BlockPool:
    """Fixed-size block allocator with an LRU of cached, unreferenced blocks.

    Every block is in exactly one of three states, which is what keeps the
    published KV events honest:

    * **referenced** -- held by at least one in-flight request (``_refcount``);
    * **free-cached** -- unreferenced but still holding a usable prefix, so it
      is reusable by a matching request and evictable under pressure;
    * **free-clean** -- holding nothing (``_free_clean``).

    Allocation takes clean blocks first and only evicts a cached one when
    nothing clean is left. Taking cached blocks while clean ones sat unused
    would throw away a live prefix on every new request -- and, since eviction
    is published, would tell the router a prefix disappeared moments after it
    appeared.

    ``on_evict`` receives the external ids of blocks that have left the cache.

    Every operation is independent of the pool size: the evictable count is
    kept as a running total, and eviction pops the least recently used
    evictable leaf from a heap keyed by access tick. A pool holds tens of
    thousands of blocks, and the scheduler queries it on every step.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        on_evict: Optional[Callable[[List[int]], None]] = None,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._on_evict = on_evict
        self._on_change = on_change
        self.evicted_blocks = 0
        self.cleared_blocks = 0
        self._free_clean: List[int] = list(range(num_blocks))
        self._free_clean_set: Set[int] = set(self._free_clean)
        self._cached: "OrderedDict[bytes, int]" = OrderedDict()  # digest -> block
        self._block_to_digest: Dict[int, bytes] = {}
        self._parents: Dict[bytes, Optional[bytes]] = {}
        self._children: Dict[bytes, int] = {}  # digest -> number of cached children
        self._refcount: Dict[int, int] = {}
        # Last-access tick per cached digest; ticks increase in ``_cached`` order.
        self._ticks: Dict[bytes, int] = {}
        self._tick = 0
        self._evictable = 0  # cached blocks with no reference
        # (tick, digest) of evictable leaves. Entries go stale when the digest is
        # touched, referenced, gains a child or leaves; ``_evict_one`` skips them.
        self._leaves: List[Tuple[int, bytes]] = []

    # -- capacity ----------------------------------------------------------

    @property
    def used_blocks(self) -> int:
        """Blocks holding content, cached ones included -- a cached block still
        occupies KV memory, which is exactly what makes it evictable."""
        return self.num_blocks - len(self._free_clean)

    @property
    def free_blocks(self) -> int:
        return len(self._free_clean)

    @property
    def cached_blocks(self) -> int:
        return len(self._cached)

    @property
    def evictable_blocks(self) -> int:
        return self._evictable_count()

    @property
    def usage(self) -> float:
        return self.used_blocks / self.num_blocks if self.num_blocks else 0.0

    def can_allocate(self, count: int) -> bool:
        return count <= len(self._free_clean) + self._evictable_count()

    # -- allocation --------------------------------------------------------

    def allocate(self, count: int) -> Optional[List[int]]:
        """Take ``count`` blocks: clean first, then LRU eviction."""
        if count <= 0:
            return []

        blocks: List[int] = []
        evicted: List[int] = []
        for _ in range(count):
            if not self._free_clean and not self._evict_one(collect=evicted):
                # Roll back so a failed admission leaves no blocks stranded.
                self._release(blocks)
                self._notify_evicted(evicted)
                return None
            block = self._free_clean.pop()
            self._free_clean_set.discard(block)
            self._ref(block)
            blocks.append(block)

        self._notify_evicted(evicted)
        return blocks

    def free(self, blocks: Sequence[int]) -> None:
        for block in blocks:
            remaining = self._refcount.get(block, 1) - 1
            if remaining > 0:
                self._refcount[block] = remaining
                continue
            if self._refcount.pop(block, 0) > 0:
                self._unreferenced(block)
            # A block still holding a cached prefix stays reusable rather than
            # going back to the clean pool; only eviction clears it.
            if block not in self._block_to_digest:
                self._add_clean(block)
        self._notify_changed()

    def _release(self, blocks: Sequence[int]) -> None:
        for block in blocks:
            if self._refcount.pop(block, 0) > 0:
                self._unreferenced(block)
            self._add_clean(block)

    # -- prefix cache ------------------------------------------------------

    def match_prefix(self, hashes: Sequence[bytes]) -> int:
        """Number of leading blocks already resident. Stops at the first miss:
        a prefix cache is only useful contiguously from the start."""
        matched = 0
        for digest in hashes:
            if digest not in self._cached:
                break
            self._touch(digest)
            matched += 1
        return matched

    def acquire_prefix(self, hashes: Sequence[bytes]) -> List[int]:
        """Match a prefix and take a reference on every block it hits.

        Sharing the blocks is what makes a cache hit real: the request pays no
        memory for the matched prefix, and the reference stops those blocks
        being evicted underneath it. Matching without acquiring would report a
        hit and then let the next allocation invalidate it.
        """
        acquired: List[int] = []
        for digest in hashes:
            block = self._cached.get(digest)
            if block is None:
                break
            self._touch(digest)
            self._ref(block)
            acquired.append(block)
        self._notify_changed()
        return acquired

    def insert_prefix(
        self, hashes: Sequence[bytes], blocks: Sequence[int]
    ) -> List[Tuple[int, bytes]]:
        """Publish blocks into the prefix cache.

        Returns ``(index, digest)`` for entries that were **newly** added. A
        block already cached is not re-reported: the router has it, and a
        duplicate store would be redundant traffic at best and a double-count
        at worst.
        """
        added: List[Tuple[int, bytes]] = []
        for index, (digest, block) in enumerate(zip(hashes, blocks)):
            if digest in self._cached:
                self._touch(digest)
                continue
            # A block can only carry one prefix; re-keying it would strand the
            # old digest pointing at content that is no longer there.
            previous = self._block_to_digest.get(block)
            if previous is not None and previous != digest:
                continue
            parent = hashes[index - 1] if index else None
            self._cached[digest] = block
            self._block_to_digest[block] = digest
            self._parents[digest] = parent
            if parent is not None:
                self._children[parent] = self._children.get(parent, 0) + 1
            self._tick += 1
            self._ticks[digest] = self._tick
            if self._refcount.get(block, 0) == 0:
                self._evictable += 1
                self._push_leaf(digest)
            added.append((index, digest))
        self._notify_changed()
        return added

    def clear_cache(self) -> None:
        """Drop every unreferenced cache entry, reporting the removals."""
        evicted: List[int] = []
        while self._evict_one(collect=evicted, clearing=True):
            pass
        self._notify_evicted(evicted)

    # -- internals ---------------------------------------------------------

    def _evictable_count(self) -> int:
        return self._evictable

    def _evict_one(self, collect: Optional[List[int]] = None, *, clearing: bool = False) -> bool:
        """Evict an LRU leaf so no reusable child outlives its published parent."""
        while self._leaves:
            tick, digest = heapq.heappop(self._leaves)
            if self._ticks.get(digest) != tick or not self._is_evictable_leaf(digest):
                continue
            block = self._cached.pop(digest)
            del self._ticks[digest]
            parent = self._parents.pop(digest)
            self._block_to_digest.pop(block, None)
            self._evictable -= 1
            if parent is not None:
                remaining = self._children[parent] - 1
                if remaining:
                    self._children[parent] = remaining
                else:
                    del self._children[parent]
                    self._push_leaf(parent)
            if clearing:
                self.cleared_blocks += 1
            else:
                self.evicted_blocks += 1
            self._add_clean(block)
            if collect is None:
                self._notify_evicted([external_id(digest)])
            else:
                collect.append(external_id(digest))
            return True
        return False

    def _is_evictable_leaf(self, digest: bytes) -> bool:
        block = self._cached.get(digest)
        return (
            block is not None
            and self._refcount.get(block, 0) == 0
            and digest not in self._children
        )

    def _push_leaf(self, digest: bytes) -> None:
        if not self._is_evictable_leaf(digest):
            return
        heapq.heappush(self._leaves, (self._ticks[digest], digest))
        if len(self._leaves) > 2 * len(self._cached) + 64:
            self._leaves = [
                (tick, d) for d, tick in self._ticks.items() if self._is_evictable_leaf(d)
            ]
            heapq.heapify(self._leaves)

    def _touch(self, digest: bytes) -> None:
        self._cached.move_to_end(digest)
        self._tick += 1
        self._ticks[digest] = self._tick
        self._push_leaf(digest)

    def _ref(self, block: int) -> None:
        count = self._refcount.get(block, 0)
        self._refcount[block] = count + 1
        if count == 0 and block in self._block_to_digest:
            self._evictable -= 1

    def _unreferenced(self, block: int) -> None:
        digest = self._block_to_digest.get(block)
        if digest is not None:
            self._evictable += 1
            self._push_leaf(digest)

    def _add_clean(self, block: int) -> None:
        if block not in self._free_clean_set:
            self._free_clean_set.add(block)
            self._free_clean.append(block)

    def _notify_evicted(self, ids: List[int]) -> None:
        self._notify_changed()
        if ids and self._on_evict is not None:
            self._on_evict(ids)

    def _notify_changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
