#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def read_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_dict(orient="records")
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("json input must be a list")
        return data
    raise ValueError(f"Unsupported input file: {path}")


def extract_field(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def to_verl_row(question: str, answer: str, data_source: str, split: str, idx: int, instruction: str) -> dict[str, Any]:
    question_text = f"{question}\n\n{instruction}".strip()
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question_text}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "raw_question": question,
            "raw_answer": answer,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare verl parquet dataset from raw data.")
    parser.add_argument("--input", type=str, required=True, help="json/jsonl/parquet input")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="ttsr_custom")
    parser.add_argument("--question_keys", type=str, default="question,problem")
    parser.add_argument("--answer_keys", type=str, default="answer,solution,ground_truth")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--instruction",
        type=str,
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    question_keys = [x.strip() for x in args.question_keys.split(",") if x.strip()]
    answer_keys = [x.strip() for x in args.answer_keys.split(",") if x.strip()]

    raw_rows = read_rows(args.input)
    converted: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        question = extract_field(row, question_keys)
        if not question:
            continue
        answer = extract_field(row, answer_keys)
        converted.append(
            to_verl_row(
                question=question,
                answer=answer,
                data_source=args.data_source,
                split="raw",
                idx=i,
                instruction=args.instruction,
            )
        )

    if not converted:
        raise ValueError("No valid rows found after field mapping.")

    rng.shuffle(converted)
    train_size = int(len(converted) * max(0.0, min(1.0, args.train_ratio)))
    train_rows = converted[:train_size]
    test_rows = converted[train_size:] or converted[-1:]

    # rewrite split/index
    for i, row in enumerate(train_rows):
        row["extra_info"]["split"] = "train"
        row["extra_info"]["index"] = i
    for i, row in enumerate(test_rows):
        row["extra_info"]["split"] = "test"
        row["extra_info"]["index"] = i

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(test_rows).to_parquet(test_path)
    print(f"[prepare_dataset] train={len(train_rows)} -> {train_path}")
    print(f"[prepare_dataset] test={len(test_rows)} -> {test_path}")


if __name__ == "__main__":
    main()

