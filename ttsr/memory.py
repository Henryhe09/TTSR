import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.split(), b.split()).ratio()


@dataclass
class WeaknessEntry:
    reasoning_weakness: str
    trigger_conditions: list[str]
    failure_signature: list[str]
    localization_summary: str
    first_seen: int
    last_seen: int
    persistence: int = 1

    @classmethod
    def from_reflection(cls, item: dict[str, Any], iteration: int) -> "WeaknessEntry":
        return cls(
            reasoning_weakness=str(item.get("reasoning_weakness", "")).strip(),
            trigger_conditions=[str(x) for x in item.get("trigger_conditions", []) if str(x).strip()],
            failure_signature=[str(x) for x in item.get("failure_signature", []) if str(x).strip()],
            localization_summary=str(item.get("localization_summary", "")).strip(),
            first_seen=iteration,
            last_seen=iteration,
            persistence=1,
        )


class WeaknessMemory:
    def __init__(self, capacity: int = 10, merge_threshold: float = 0.6, resolve_window: int = 3):
        self.capacity = int(capacity)
        self.merge_threshold = float(merge_threshold)
        self.resolve_window = int(resolve_window)
        self.entries: list[WeaknessEntry] = []

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.entries = [WeaknessEntry(**x) for x in raw]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(x) for x in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update(self, reflections: list[dict[str, Any]], iteration: int) -> None:
        for item in reflections:
            candidate = WeaknessEntry.from_reflection(item, iteration)
            if not candidate.reasoning_weakness:
                continue
            matched = False
            for existing in self.entries:
                sim = _similarity(candidate.reasoning_weakness, existing.reasoning_weakness)
                if sim >= self.merge_threshold:
                    existing.persistence += 1
                    existing.last_seen = iteration
                    matched = True
                    break
            if not matched:
                self.entries.append(candidate)

        self.entries = [x for x in self.entries if iteration - x.last_seen <= self.resolve_window]
        self.entries.sort(key=lambda x: x.persistence, reverse=True)
        self.entries = self.entries[: self.capacity]

    def top(self, n: int = 3) -> list[WeaknessEntry]:
        return sorted(self.entries, key=lambda x: x.persistence, reverse=True)[:n]

    def strategy_note(self, n: int = 3) -> str:
        picks = self.top(n)
        if not picks:
            return ""
        lines: list[str] = []
        for idx, item in enumerate(picks, 1):
            trigger = ", ".join(item.trigger_conditions) if item.trigger_conditions else "general"
            lines.append(
                f"{idx}. {item.reasoning_weakness} (trigger: {trigger}; seen {item.persistence} rounds)"
            )
        return "\n".join(lines)

    def synthesis_context(self, n: int = 5) -> str:
        picks = self.top(n)
        if not picks:
            return "None."
        lines: list[str] = []
        for idx, item in enumerate(picks, 1):
            lines.append(f"{idx}. {item.reasoning_weakness}")
        return "\n".join(lines)

