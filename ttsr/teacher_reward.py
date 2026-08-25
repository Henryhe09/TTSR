"""Teacher GRPO reward driven by online Student pseudo-correctness."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from ttsr.io_utils import try_parse_json_object

QUESTION_TAG_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)


@dataclass
class _PendingTeacherItem:
    generated_question: str
    reference_question: str
    format_ok: bool
    future: asyncio.Future


def token_similarity(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a.split(), b.split()).ratio())


def extract_generated_question(text: str) -> tuple[str, bool]:
    text = (text or "").strip()
    if not text:
        return "", False
    obj = try_parse_json_object(text)
    if isinstance(obj, dict):
        candidate = str(obj.get("generated_question", "")).strip()
        if candidate:
            matches = QUESTION_TAG_RE.findall(candidate)
            return (matches[-1].strip(), True) if matches else (candidate, True)
    matches = QUESTION_TAG_RE.findall(text)
    return (matches[-1].strip(), True) if matches and matches[-1].strip() else ("", False)


def evaluate_questions(url: str, questions: list[str], timeout_s: float) -> list[dict[str, Any]]:
    request = Request(
        url.rstrip("/") + "/evaluate",
        data=json.dumps({"questions": questions}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # nosec B310: URL is explicit experiment config
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Student evaluator unavailable at {url}: {exc}") from exc
    results = payload.get("results", [])
    if not isinstance(results, list) or len(results) != len(questions):
        raise RuntimeError("Student evaluator returned malformed or incomplete results")
    return results


class TTSRStudentFrontierTeacherRewardManager(RewardManagerBase):
    """Teacher reward R_T=max(0, 4s(1-s)-lambda*R_sim).

    ``s`` is evaluated from G rollouts of the frozen current Student through a
    dedicated service.  A missing evaluator yields zero reward rather than a
    lexical proxy, so experiments cannot silently deviate from the paper.
    """

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config=config, tokenizer=tokenizer, compute_score=compute_score)
        options = config.reward.get("reward_kwargs", {})
        self.group_wait_ms = int(options.get("group_wait_ms", 1000))
        self.result_timeout_s = float(options.get("result_timeout_s", 300.0))
        self.student_eval_url = str(options.get("student_eval_url", "")).strip()
        self.student_eval_timeout_s = float(options.get("student_eval_timeout_s", 240.0))
        self.ttsr_lambda = float(options.get("ttsr_lambda", 1.0))
        self.ttsr_tau = float(options.get("ttsr_tau", 0.75))
        self.expected_group_size = max(1, int(options.get("expected_group_size", 1)))
        if not self.student_eval_url:
            raise ValueError("student_eval_url is required for the paper-faithful Teacher reward")
        self._items: dict[str, list[_PendingTeacherItem]] = defaultdict(list)
        self._timer_set: set[str] = set()
        self._lock = asyncio.Lock()

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        assert len(data) == 1, "Teacher reward expects one rollout at a time."
        item = data[0]
        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = item.batch["attention_mask"][-response_length:].sum()
        text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids[:valid_length], skip_special_tokens=True)
        )
        question, format_ok = extract_generated_question(text)
        extra_info = item.non_tensor_batch.get("extra_info", {}) or {}
        reference = str(extra_info.get("reference_question", "")).strip()
        uid = str(item.non_tensor_batch.get("uid", "")) or f"fallback::{hash(reference)}"
        future = self.loop.create_future()
        pending = _PendingTeacherItem(question, reference, format_ok, future)

        async with self._lock:
            group = self._items[uid]
            group.append(pending)
            if len(group) >= self.expected_group_size:
                self.loop.call_soon(lambda gid=uid: asyncio.create_task(self._finalize_group(gid)))
            elif uid not in self._timer_set:
                self._timer_set.add(uid)
                self.loop.call_later(
                    max(0.001, self.group_wait_ms / 1000.0),
                    lambda gid=uid: asyncio.create_task(self._finalize_group(gid)),
                )

        try:
            return await asyncio.wait_for(future, timeout=self.result_timeout_s)
        except asyncio.TimeoutError:
            return self._zero(uid, question, reference, format_ok, "reward timeout")

    def _zero(self, uid: str, question: str, reference: str, format_ok: bool, error: str = "") -> dict[str, Any]:
        return {
            "reward_score": 0.0,
            "reward_extra_info": {
                "group_uid": uid,
                "generated_question": question,
                "reference_question": reference,
                "format_ok": int(format_ok),
                "pseudo_correctness": 0.0,
                "frontier_reward": 0.0,
                "r_sim": 0.0,
                "acc": 0.0,
                "evaluator_error": error,
            },
        }

    async def _finalize_group(self, uid: str) -> None:
        async with self._lock:
            items = self._items.pop(uid, [])
            self._timer_set.discard(uid)
        if not items:
            return

        valid_questions = [item.generated_question for item in items if item.format_ok and item.generated_question]
        try:
            evaluations = await self.loop.run_in_executor(
                None, lambda: evaluate_questions(self.student_eval_url, valid_questions, self.student_eval_timeout_s)
            ) if valid_questions else []
            eval_iter = iter(evaluations)
            per_item_eval = [next(eval_iter) if item.format_ok and item.generated_question else None for item in items]
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_result(self._zero(uid, item.generated_question, item.reference_question, item.format_ok, str(exc)))
            return

        questions = [item.generated_question for item in items]
        for i, (item, evaluation) in enumerate(zip(items, per_item_eval, strict=True)):
            if evaluation is None:
                result = self._zero(uid, item.generated_question, item.reference_question, False, "invalid <question> format")
            else:
                score = float(evaluation.get("pseudo_correctness", 0.0))
                frontier = 4.0 * score * (1.0 - score)
                similarities = [max(0.0, token_similarity(item.generated_question, item.reference_question) - self.ttsr_tau)]
                similarities.extend(
                    max(0.0, token_similarity(item.generated_question, other) - self.ttsr_tau)
                    for j, other in enumerate(questions)
                    if i != j and other.strip()
                )
                r_sim = sum(similarities) / len(similarities)
                reward = max(0.0, frontier - self.ttsr_lambda * r_sim)
                result = {
                    "reward_score": reward,
                    "reward_extra_info": {
                        "group_uid": uid,
                        "generated_question": item.generated_question,
                        "reference_question": item.reference_question,
                        "format_ok": 1,
                        "group_size": len(items),
                        "student_rollout_n": int(evaluation.get("rollout_n", 0)),
                        "pseudo_target": str(evaluation.get("pseudo_target", "")),
                        "pseudo_correctness": score,
                        "frontier_reward": frontier,
                        "r_sim": r_sim,
                        "tau": self.ttsr_tau,
                        "lambda": self.ttsr_lambda,
                        "acc": reward,
                    },
                }
            if not item.future.done():
                item.future.set_result(result)
