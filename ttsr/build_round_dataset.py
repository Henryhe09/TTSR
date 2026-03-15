import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ttsr.io_utils import read_jsonl
from ttsr.prompts import student_prompt


def _load_parquet_rows(path: str) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict(orient="records")


def _extract_question_from_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt", None)
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    extra = row.get("extra_info", {})
    if isinstance(extra, dict) and isinstance(extra.get("raw_question"), str):
        return extra["raw_question"]
    return ""


def _as_prompt_row(question: str, strategy_note: str, source: str, idx: int, is_synthetic: bool) -> dict[str, Any]:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": student_prompt(question=question, strategy_note=strategy_note)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": idx,
            "raw_question": question,
            "is_synthetic": is_synthetic,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build next-round train parquet from base data + synthesized variants.")
    parser.add_argument("--base_train", type=str, required=True)
    parser.add_argument("--variants", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--strategy_note", type=str, default="")
    parser.add_argument("--real_ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_rows = _load_parquet_rows(args.base_train)
    variant_rows = [x for x in read_jsonl(args.variants) if str(x.get("question", "")).strip()]

    strategy_note = ""
    if args.strategy_note and Path(args.strategy_note).exists():
        strategy_note = Path(args.strategy_note).read_text(encoding="utf-8")

    n_syn = len(variant_rows)
    n_real = int(max(1, round(args.real_ratio * max(1, n_syn))))
    n_real = min(n_real, len(base_rows))
    picked_real = rng.sample(base_rows, n_real) if n_real > 0 else []

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(variant_rows):
        question = str(row.get("question", "")).strip()
        if question:
            out_rows.append(_as_prompt_row(question, strategy_note, "ttsr_variant", i, True))

    for i, row in enumerate(picked_real):
        question = _extract_question_from_row(row).strip()
        if question:
            out_rows.append(_as_prompt_row(question, strategy_note, "ttsr_base", i, False))

    # de-duplicate exact question text
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for row in out_rows:
        question = str(row["extra_info"]["raw_question"])
        if question in seen:
            continue
        seen.add(question)
        dedup.append(row)

    rng.shuffle(dedup)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dedup).to_parquet(out_path)
    print(f"[build_round_dataset] variants={n_syn} real={n_real} total={len(dedup)} -> {out_path}")


if __name__ == "__main__":
    main()

