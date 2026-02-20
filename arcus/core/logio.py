from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    steps: int
    stress: str
    competence: float
    coherence: float
    continuity: float
    identity: float
    collapse: bool
    notes: str = ""


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any] | EpisodeLog]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, EpisodeLog):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
