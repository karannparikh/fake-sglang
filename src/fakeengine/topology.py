"""Network distance between workers, for distance-dependent KV transfer.

A *location* is a path such as ``region-a/cluster-1/zone-2``, one component per
topology level (outermost first). Two workers are as close as the deepest level
their paths share, and that level names the *link* a KV transfer between them
crosses:

    levels: [region, cluster, zone]
    links:
      zone:    same zone                    (all three components equal)
      cluster: same cluster, other zone     (region and cluster equal)
      region:  same region, other cluster   (region equal)
      default: nothing shared

Each link has its own bandwidth and fixed latency, replacing the single global
``--fake-kv-bandwidth-gb-s`` / ``--fake-kv-transfer-overhead-ms`` for transfers
where both ends have a location. The topology arrives as one JSON document
(``--fake-topology``) so every worker in a launch shares the same table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

DEFAULT_LINK = "default"


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class Link:
    name: str
    bandwidth_gb_s: float
    latency_ms: float


@dataclass(frozen=True)
class Topology:
    levels: Tuple[str, ...]
    links: Mapping[str, Link]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Topology":
        levels = tuple(str(level) for level in raw.get("levels") or ())
        if not levels:
            raise TopologyError("topology.levels must list at least one level")
        if len(set(levels)) != len(levels) or DEFAULT_LINK in levels:
            raise TopologyError(
                f"topology.levels must be unique and must not use {DEFAULT_LINK!r}: "
                f"{list(levels)}"
            )

        raw_links = raw.get("links") or {}
        expected = set(levels) | {DEFAULT_LINK}
        missing = sorted(expected - set(raw_links))
        unknown = sorted(set(raw_links) - expected)
        if missing or unknown:
            raise TopologyError(
                f"topology.links needs exactly {sorted(expected)}; "
                f"missing {missing}, unknown {unknown}"
            )

        links: Dict[str, Link] = {}
        for name, spec in raw_links.items():
            try:
                bandwidth = float(spec["bandwidth_gb_s"])
                latency = float(spec.get("latency_ms", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise TopologyError(
                    f"topology.links.{name} needs bandwidth_gb_s (and optional "
                    f"latency_ms): {spec!r}"
                ) from exc
            if bandwidth <= 0 or latency < 0:
                raise TopologyError(
                    f"topology.links.{name}: bandwidth must be > 0 and latency >= 0"
                )
            links[name] = Link(name, bandwidth, latency)
        return cls(levels, links)

    @classmethod
    def from_json(cls, text: str) -> "Topology":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TopologyError(f"--fake-topology is not valid JSON: {exc}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": list(self.levels),
            "links": {
                name: {"bandwidth_gb_s": link.bandwidth_gb_s, "latency_ms": link.latency_ms}
                for name, link in self.links.items()
            },
        }

    def parse_location(self, location: str) -> List[str]:
        parts = [part for part in str(location).split("/")]
        if len(parts) != len(self.levels) or not all(parts):
            raise TopologyError(
                f"location {location!r} must have {len(self.levels)} non-empty "
                f"components ({'/'.join(self.levels)})"
            )
        return parts

    def link_between(self, a: str, b: str) -> Link:
        pa, pb = self.parse_location(a), self.parse_location(b)
        shared = 0
        for x, y in zip(pa, pb):
            if x != y:
                break
            shared += 1
        if shared == 0:
            return self.links[DEFAULT_LINK]
        return self.links[self.levels[shared - 1]]


def load(text: Optional[str]) -> Optional[Topology]:
    return Topology.from_json(text) if text else None
