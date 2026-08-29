from __future__ import annotations
from dataclasses import asdict, dataclass


def _positive(value: int, name: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    weight_bytes: int
    context: int
    layers: int
    kv_heads: int
    head_dim: int
    kv_element_bytes: int = 2
    runtime_overhead_bytes: int = 0
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("job name must be a non-empty string")
        for field in ("weight_bytes", "context", "layers", "kv_heads", "head_dim", "kv_element_bytes"):
            _positive(getattr(self, field), field)
        _positive(self.runtime_overhead_bytes, "runtime_overhead_bytes", zero=True)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")

    @property
    def kv_cache_bytes(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.context * self.kv_element_bytes

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_cache_bytes + self.runtime_overhead_bytes


@dataclass(frozen=True, slots=True)
class Decision:
    name: str
    accepted: bool
    required_bytes: int
    weight_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int
    remaining_after_bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class Plan:
    ram_bytes: int
    reserve_bytes: int
    usable_bytes: int
    admitted_bytes: int
    remaining_bytes: int
    decisions: tuple[Decision, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decisions"] = [asdict(item) for item in self.decisions]
        return payload


def plan_jobs(ram_bytes: int, reserve_bytes: int, jobs: list[Job]) -> Plan:
    _positive(ram_bytes, "ram_bytes")
    _positive(reserve_bytes, "reserve_bytes", zero=True)
    if reserve_bytes >= ram_bytes:
        raise ValueError("reserve_bytes must be smaller than ram_bytes")
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("job names must be unique")
    usable = ram_bytes - reserve_bytes
    remaining = usable
    decisions: list[Decision] = []
    indexed = sorted(enumerate(jobs), key=lambda pair: (-pair[1].priority, pair[0]))
    for _, job in indexed:
        accepted = job.total_bytes <= remaining
        if accepted:
            remaining -= job.total_bytes
        decisions.append(Decision(
            name=job.name, accepted=accepted, required_bytes=job.total_bytes,
            weight_bytes=job.weight_bytes, kv_cache_bytes=job.kv_cache_bytes,
            runtime_overhead_bytes=job.runtime_overhead_bytes,
            remaining_after_bytes=remaining,
            reason="admitted" if accepted else f"short by {job.total_bytes - remaining} bytes",
        ))
    return Plan(ram_bytes, reserve_bytes, usable, usable - remaining, remaining, tuple(decisions))
