"""HTTP service that measures TTSR Student pseudo-correctness on generated variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ttsr.io_utils import extract_final_answer
from ttsr.prompts import student_prompt


class StudentEvaluator:
    def __init__(
        self,
        model: str,
        strategy_note: str,
        rollout_n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
    ) -> None:
        try:
            import vllm
        except ModuleNotFoundError as exc:
            raise RuntimeError("vllm is required for Student evaluation") from exc
        self.vllm = vllm
        self.strategy_note = strategy_note
        self.rollout_n = rollout_n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.llm = vllm.LLM(
            model=model,
            tokenizer=model,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

    def evaluate(self, questions: list[str]) -> list[dict[str, Any]]:
        prompts = [student_prompt(question=question, strategy_note=self.strategy_note) for question in questions]
        params = self.vllm.SamplingParams(
            n=self.rollout_n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        results: list[dict[str, Any]] = []
        for question, output in zip(questions, outputs, strict=True):
            traces = [candidate.text for candidate in output.outputs]
            answers = [extract_final_answer(trace) for trace in traces]
            nonempty = [answer.strip() for answer in answers if answer.strip()]
            if nonempty:
                counts = Counter(nonempty)
                pseudo_target, majority_count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
            else:
                pseudo_target, majority_count = "", 0
            rewards = [int(bool(pseudo_target) and answer.strip() == pseudo_target) for answer in answers]
            score = float(sum(rewards) / len(rewards)) if rewards else 0.0
            results.append(
                {
                    "question": question,
                    "rollout_n": len(traces),
                    "traces": traces,
                    "answers": answers,
                    "pseudo_target": pseudo_target,
                    "rewards": rewards,
                    "pseudo_correctness": score,
                    "frontier_reward": 4.0 * score * (1.0 - score),
                    "majority_count": majority_count,
                }
            )
        return results


def make_handler(evaluator: StudentEvaluator):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, {"ok": True})
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/evaluate":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                questions = payload.get("questions", [])
                if not isinstance(questions, list) or not questions or not all(isinstance(q, str) and q.strip() for q in questions):
                    raise ValueError("questions must be a non-empty list of strings")
                self._write_json(HTTPStatus.OK, {"results": evaluator.evaluate(questions)})
            except Exception as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Student rollout evaluations for paper-faithful TTSR.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy_note", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--rollout_n", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    strategy = Path(args.strategy_note).read_text(encoding="utf-8") if args.strategy_note else ""
    evaluator = StudentEvaluator(
        model=args.model,
        strategy_note=strategy,
        rollout_n=args.rollout_n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(evaluator))
    print(f"[student_evaluator] ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
