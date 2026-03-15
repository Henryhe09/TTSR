import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_RE = re.compile(r"(?i)answer\s*:\s*([^\n]+)")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid json at {path}:{line_no}") from exc
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right <= left:
        return None
    try:
        obj = json.loads(text[left : right + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    answer = ANSWER_RE.findall(text)
    if answer:
        return answer[-1].strip()

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return lines[-1][:256]

