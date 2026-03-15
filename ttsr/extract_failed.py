import argparse
from pathlib import Path
from typing import Any

from ttsr.io_utils import read_jsonl, write_jsonl


def _iter_rollout_rows(rollout_dir: Path):
    for path in sorted(rollout_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            yield path, row


def _extract_question(prompt_text: str) -> str:
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        return ""
    marker = "Question:"
    if marker in prompt_text:
        return prompt_text.split(marker, 1)[-1].strip()
    return prompt_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract failed trajectories from verl rollout dumps.")
    parser.add_argument("--rollout_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--reward_threshold", type=float, default=0.0)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        raise FileNotFoundError(f"rollout_dir not found: {rollout_dir}")

    failed: list[dict[str, Any]] = []
    for path, row in _iter_rollout_rows(rollout_dir):
        score = float(row.get("score", 0.0))
        if score > args.reward_threshold:
            continue
        uid = str(row.get("group_uid", "")) or f"{path.stem}:{len(failed)}"
        prompt = str(row.get("input", ""))
        output = str(row.get("output", ""))
        pseudo_target = str(row.get("pseudo_target", ""))
        failed.append(
            {
                "uid": uid,
                "question": _extract_question(prompt),
                "failed_trace": output,
                "pseudo_target": pseudo_target,
                "score": score,
                "source_file": str(path),
            }
        )

    failed = failed[: max(0, args.max_samples)]
    write_jsonl(args.out, failed)
    print(f"[extract_failed] saved {len(failed)} rows -> {args.out}")


if __name__ == "__main__":
    main()

