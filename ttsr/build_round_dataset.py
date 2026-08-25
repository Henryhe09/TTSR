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
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    extra = row.get("extra_info", {})
    return str(extra.get("raw_question", "")) if isinstance(extra, dict) else ""


def _as_prompt_row(question: str, strategy_note: str, source: str, idx: int, is_synthetic: bool) -> dict[str, Any]:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": student_prompt(question=question, strategy_note=strategy_note)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": idx, "raw_question": question, "is_synthetic": is_synthetic},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D_t = X_test union X_var for the next TTSR iteration.")
    parser.add_argument("--base_train", required=True, help="The complete original test-question parquet X_test.")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy_note", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_rows = _load_parquet_rows(args.base_train)
    variants = [row for row in read_jsonl(args.variants) if str(row.get("question", "")).strip()]
    strategy_note = Path(args.strategy_note).read_text(encoding="utf-8") if args.strategy_note and Path(args.strategy_note).exists() else ""

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(base_rows):
        question = _extract_question_from_row(row).strip()
        if question:
            rows.append(_as_prompt_row(question, strategy_note, "ttsr_test", idx, False))
    offset = len(rows)
    for idx, row in enumerate(variants):
        rows.append(_as_prompt_row(str(row["question"]).strip(), strategy_note, "ttsr_variant", offset + idx, True))

    seen: set[str] = set()
    deduped = [row for row in rows if not (row["extra_info"]["raw_question"] in seen or seen.add(row["extra_info"]["raw_question"]))]
    random.Random(args.seed).shuffle(deduped)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(deduped).to_parquet(out)
    print(f"[build_round_dataset] test={len(base_rows)} variants={len(variants)} total={len(deduped)} -> {out}")


if __name__ == "__main__":
    main()
