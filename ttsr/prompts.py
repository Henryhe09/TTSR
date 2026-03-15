from dataclasses import dataclass


REFLECTION_TEMPLATE = """[ROLE: TEACHER]

You are given a failed reasoning trajectory from a student model.
Analyze process-level reasoning defects only.

[QUESTION]
{question}

[FAILED_REASONING]
{failed_trace}

Output JSON only:
{{
  "reasoning_weakness": "...",
  "trigger_conditions": ["..."],
  "failure_signature": ["..."],
  "localization_summary": "..."
}}
"""


SYNTHESIS_TEMPLATE = """[ROLE: TEACHER]

Generate one new math training question targeting a known weakness.

[QUESTION]
{question}

[FAILED_REASONING]
{failed_trace}

[WEAKNESS]
{weakness_json}

[PERSISTENT_WEAKNESSES]
{persistent_weaknesses}

Output JSON only:
{{
  "generated_question": "<question>...</question>",
  "hit_rationale": ["...", "..."]
}}
"""


STUDENT_TEMPLATE = """{strategy_note}
Question:
{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def reflection_prompt(question: str, failed_trace: str) -> str:
    return REFLECTION_TEMPLATE.format(question=question, failed_trace=failed_trace)


def synthesis_prompt(
    question: str,
    failed_trace: str,
    weakness_json: str,
    persistent_weaknesses: str,
) -> str:
    return SYNTHESIS_TEMPLATE.format(
        question=question,
        failed_trace=failed_trace,
        weakness_json=weakness_json,
        persistent_weaknesses=persistent_weaknesses,
    )


def student_prompt(question: str, strategy_note: str = "") -> str:
    prefix = strategy_note.strip()
    if prefix:
        prefix = f"[REASONING_STRATEGY_NOTES]\n{prefix}\n\n"
    return STUDENT_TEMPLATE.format(strategy_note=prefix, question=question)


@dataclass
class PromptPack:
    reflection: str
    synthesis: str
    student: str

