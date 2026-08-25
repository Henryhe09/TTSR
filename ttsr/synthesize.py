import argparse
import json
import re
from pathlib import Path
from typing import Any

from ttsr.io_utils import read_jsonl, try_parse_json_object, write_jsonl
from ttsr.memory import WeaknessMemory
from ttsr.prompts import synthesis_prompt
from ttsr.teacher_reward import evaluate_questions


QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)


def _extract_question(text: str, allow_plain: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    matches = QUESTION_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text if allow_plain else ""

def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher synthesis for TTSR variants.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--failed_instances", type=str, required=True)
    parser.add_argument("--memory", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--reflections", type=str, default="")
    parser.add_argument("--max_variants", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student_eval_url", type=str, required=True)
    parser.add_argument("--student_eval_timeout_s", type=float, default=240.0)
    parser.add_argument("--rollout_cache", type=str, required=True)
    args = parser.parse_args()

    try:
        import vllm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "vllm is required for synthesis. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    failed = list(read_jsonl(args.failed_instances))
    if args.max_variants > 0:
        failed = failed[: args.max_variants]

    uid_to_reflection: dict[str, dict[str, Any]] = {}
    if args.reflections and Path(args.reflections).exists():
        for row in read_jsonl(args.reflections):
            uid = str(row.get("uid", ""))
            reflection = row.get("reflection")
            if uid and isinstance(reflection, dict):
                uid_to_reflection[uid] = reflection

    memory = WeaknessMemory()
    memory.load(args.memory)
    persistent = memory.synthesis_context()

    prompts: list[str] = []
    for row in failed:
        uid = str(row.get("uid", ""))
        reflection = uid_to_reflection.get(uid, {})
        prompts.append(
            synthesis_prompt(
                question=str(row.get("question", "")),
                failed_trace=str(row.get("failed_trace", "")),
                weakness_json=json.dumps(reflection, ensure_ascii=False),
                persistent_weaknesses=persistent,
            )
        )

    llm = vllm.LLM(model=args.model, tokenizer=args.model, seed=args.seed, gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    rows: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for src, out in zip(failed, outputs, strict=False):
        raw = out.outputs[0].text if out.outputs else ""
        obj = try_parse_json_object(raw)
        q = ""
        if obj:
            q = _extract_question(str(obj.get("generated_question", "")), allow_plain=True)
        if not q:
            q = _extract_question(raw)

        rows.append(
            {
                "source_uid": src.get("uid", ""),
                "reference_question": src.get("question", ""),
                "question": q,
                "is_synthetic": True,
                "teacher_raw": raw,
            }
        )
        debug.append({"source_uid": src.get("uid", ""), "parsed_ok": bool(q), "raw": raw})

    valid_rows = [row for row in rows if str(row.get("question", "")).strip()]
    evaluations = evaluate_questions(args.student_eval_url, [str(row["question"]) for row in valid_rows], args.student_eval_timeout_s)
    for row, evaluation in zip(valid_rows, evaluations, strict=True):
        row["student_evaluation"] = evaluation
    write_jsonl(args.out, rows)
    write_jsonl(args.rollout_cache, valid_rows)
    write_jsonl(Path(args.out).with_suffix(".debug.jsonl"), debug)
    print(f"[synthesize] saved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
