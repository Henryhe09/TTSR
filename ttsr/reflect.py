import argparse
from pathlib import Path
from typing import Any

from ttsr.io_utils import read_jsonl, try_parse_json_object, write_jsonl
from ttsr.memory import WeaknessMemory
from ttsr.prompts import reflection_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher reflection + weakness memory update.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--failed_instances", type=str, required=True)
    parser.add_argument("--memory_out", type=str, required=True)
    parser.add_argument("--strategy_out", type=str, required=True)
    parser.add_argument("--reflections_out", type=str, required=True)
    parser.add_argument("--memory_in", type=str, default="")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import vllm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "vllm is required for reflection. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    failed_instances = list(read_jsonl(args.failed_instances))
    prompts = [
        reflection_prompt(question=str(x.get("question", "")), failed_trace=str(x.get("failed_trace", "")))
        for x in failed_instances
    ]

    llm = vllm.LLM(model=args.model, tokenizer=args.model, seed=args.seed, gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    reflections: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for src, out in zip(failed_instances, outputs, strict=False):
        raw = out.outputs[0].text if out.outputs else ""
        obj = try_parse_json_object(raw)
        uid = str(src.get("uid", ""))
        if obj and obj.get("reasoning_weakness"):
            reflections.append({"uid": uid, "reflection": obj})
            debug_rows.append({"uid": uid, "ok": True, "raw": raw})
        else:
            debug_rows.append({"uid": uid, "ok": False, "raw": raw})

    memory = WeaknessMemory()
    if args.memory_in:
        memory.load(args.memory_in)
    memory.update([x["reflection"] for x in reflections], iteration=args.iteration)

    memory_path = Path(args.memory_out)
    strategy_path = Path(args.strategy_out)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(memory_path)
    strategy_path.write_text(memory.strategy_note(), encoding="utf-8")

    write_jsonl(args.reflections_out, reflections)
    write_jsonl(strategy_path.with_suffix(".reflect_debug.jsonl"), debug_rows)
    print(f"[reflect] parsed={len(reflections)} / total={len(failed_instances)}")


if __name__ == "__main__":
    main()
