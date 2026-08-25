from dataclasses import dataclass


REFLECTION_TEMPLATE = """[ROLE: TEACHER]

Your task is to extract a generalizable reasoning weakness from an original question and a failed or unstable reasoning trace produced by a student model.

You must strictly base your analysis on the given inputs. Do not introduce external knowledge, do not assume access to the correct answer, and do not judge final correctness.

[ORIGINAL_QUESTION]
{question}

[FAILED_REASONING_TRACE]
{failed_trace}

[ANALYSIS GUIDELINES]
(1) Error Localization: Identify the first point where the reasoning becomes unreliable, incomplete, or invalid.
(2) Weakness Abstraction: Summarize the underlying reasoning weakness in one abstract sentence, without referring to specific values or variables.
(3) Trigger Conditions: Describe what problem structures or conditions are likely to trigger this weakness.
(4) Failure Signature: Describe typical reasoning patterns or behaviors when this weakness appears.

[OUTPUT REQUIREMENTS]
Output must be valid JSON and contain the following fields:
{{
  "reasoning_weakness": "...",
  "trigger_conditions": ["..."],
  "failure_signature": ["..."],
  "localization_summary": "..."
}}
"""


SYNTHESIS_TEMPLATE = """[ROLE: TEACHER]

Your task is to generate one new training question based on the original question, a failed reasoning trace, and an extracted reasoning weakness.

The goal is not to make the question arbitrarily harder, but to produce a question that is targeted, learnable, and close to the student's capability frontier.

You must ensure that:
(1) The new question preserves the core reasoning structure of the original question.
(2) The new question is more likely to trigger the given reasoning weakness.
(3) The new question is self-contained, solvable, and unambiguous.
(4) The new question is not a superficial paraphrase or simple numerical substitution.

[INPUT]

[ORIGINAL_QUESTION]
{question}

[FAILED_REASONING_TRACE]
{failed_trace}

[WEAKNESS_JSON]
{weakness_json}

[PERSISTENT_WEAKNESSES]
{persistent_weaknesses}

[STEP-BY-STEP INSTRUCTIONS]

Step 1: Anchor Structure.
Summarize the core reasoning structure of the original question, including key concepts, variable relationships, constraints, and the main reasoning steps required for a correct solution.
Use structural language rather than restating the original question.

Step 2: Error-Hitting Strategy.
Explicitly state how the new question will be designed to more reliably trigger the given reasoning weakness, while remaining solvable and fair.
Explain what aspects will be modified, what shortcuts are targeted, and how superficial paraphrasing is avoided.

Step 3: Generate One New Question.
Generate a single, fully self-contained question that requires multi-step reasoning and is likely to trigger the weakness under common shortcuts.
Avoid changing only numbers, swapping story contexts, or adding irrelevant complexity.

Step 4: Hit Rationale.
Briefly explain why the generated question is likely to expose the reasoning weakness and how a correct solution would avoid the shortcut.

Step 5: Self-Test Filter.
Verify whether the question is likely to trigger the weakness, remains learnable, and is not a surface-level paraphrase.
If any check fails, regenerate the question.

[OUTPUT REQUIREMENTS]
The output must be valid JSON. Do not include markdown or additional explanations.
All fields must be present and filled according to the schema below.
{{
  "anchor_structure": ["...core structural element 1...", "...core structural element 2..."],
  "error_hitting_strategy": {{
    "what_to_avoid": ["...simple variants that would not trigger the weakness..."],
    "what_to_add": ["...structural modifications to increase weakness exposure..."],
    "shortcut_to_block": ["...erroneous shortcut to be induced or blocked..."],
    "fairness_check": "...how solvability and unambiguity are ensured..."
  }},
  "generated_question": "...fully self-contained question text...",
  "hit_rationale": ["...why the question is likely to trigger the weakness...", "...what a correct reasoning process would need to check..."],
  "self_test": {{
    "likely_to_trigger_weakness": "YES/NO + brief reason",
    "learnable_frontier": "YES/NO + brief reason",
    "not_surface_paraphrase": "YES/NO + brief reason"
  }}
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
        prefix = f"[REASONING_STRATEGY_NOTES]\\n{prefix}\\n\\n"
    return STUDENT_TEMPLATE.format(strategy_note=prefix, question=question)


@dataclass
class PromptPack:
    reflection: str
    synthesis: str
    student: str
