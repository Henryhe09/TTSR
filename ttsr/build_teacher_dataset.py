import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ttsr.io_utils import read_jsonl
from ttsr.memory import WeaknessMemory
from ttsr.prompts import synthesis_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build verl parquet dataset for Teacher GRPO training (synthesis policy)."
    )
    parser.add_argument("--failed_instances", type=str, required=True)
    parser.add_argument("--reflections", type=str, default="")
    parser.add_argument("--memory", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    failed_rows = list(read_jsonl(args.failed_instances))
    if args.max_samples > 0:
        failed_rows = failed_rows[: args.max_samples]

    uid_to_reflection: dict[str, dict[str, Any]] = {}
    if args.reflections and Path(args.reflections).exists():
        for row in read_jsonl(args.reflections):
            uid = str(row.get("uid", ""))
            reflection = row.get("reflection")
            if uid and isinstance(reflection, dict):
                uid_to_reflection[uid] = reflection

    memory = WeaknessMemory()
    if args.memory and Path(args.memory).exists():
        memory.load(args.memory)
    persistent = memory.synthesis_context()

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(failed_rows):
        uid = str(row.get("uid", ""))
        question = str(row.get("question", ""))
        failed_trace = str(row.get("failed_trace", ""))
        reflection = uid_to_reflection.get(uid, {})

        prompt_text = synthesis_prompt(
            question=question,
            failed_trace=failed_trace,
            weakness_json=json.dumps(reflection, ensure_ascii=False),
            persistent_weaknesses=persistent,
        )

        out_rows.append(
            {
                "data_source": "ttsr_teacher",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "source_uid": uid,
                    "reference_question": question,
                    "failed_trace": failed_trace,
                    "weakness_json": reflection,
                },
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(out_path)
    print(f"[build_teacher_dataset] rows={len(out_rows)} -> {out_path}")


if __name__ == "__main__":
    main()

