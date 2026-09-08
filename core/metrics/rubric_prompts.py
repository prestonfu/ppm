"""Runtime rubric and reward-metric prompts."""

RUBRIC_SYSTEM_PROMPT = """You are given:
1. A mathematical problem.
2. A grading rubric consisting of numbered criteria.
3. A student solution to the problem.

Your task is to grade the student solution against the rubric.
For each statement, output only "YES" or "NO" based on whether the solution satisfies the criterion. Format your response as a numbered list, e.g. if there are n rubric items, your response should follow the format:
1. YES
2. NO
...
n. YES
"""


STATE_MATCHING_SYSTEM_PROMPT = """You are given:
1. A mathematical problem.
2. A grading rubric consisting of numbered criteria.
3. A partial rollout (reasoning trace + solution) to solving the problem.

Grading rules:
- For each rubric criterion i, output YES if *any* part of the rollout exactly satisfies criterion i.
- Answer NO if you have checked the entire rollout and are certain that the criterion is nowhere satisfied.
- Always answer NO if the rubric item describes a feature not present in the given rollout. For example, if the rubric item asks "Do the <answer> </answer> tags contain the number 3", but there are no answer tags at all, answer NO.
- Judge each criterion independently. For example, if step 2 depends on step 1, grade step 2 based solely on whether the rollout's claimed step-2 result matches the rubric, regardless of whether step 1 was correct.
- Judge whether the rubric criterion itself is correct in the rollout, not whether the reasoning that led to it was sound.
- Do not use the rubric or problem text as evidence; only quote from the rollout.

Format your response as a numbered list (one line per criterion):
1. YES
2. NO
...
n. YES
"""


STATE_MATCHING_SCAFFOLD_SYSTEM_PROMPT = """You are given:
1. A mathematical problem.
2. A grading rubric consisting of numbered criteria.
3. A partial rollout (reasoning trace + solution) to solving the problem.

Grading rules:
- For each rubric criterion i, output YES if *any* part of the rollout exactly satisfies criterion i.
- Answer NO if you have checked the entire rollout and are certain that the criterion is nowhere satisfied.
- Always answer NO if the rubric item describes a feature not present in the given rollout. For example, if the rubric item asks "Do the <answer> </answer> tags contain the number 3", but there are no answer tags at all, answer NO.
- Judge each criterion independently. For example, if step 2 depends on step 1, grade step 2 based solely on whether the rollout's claimed step-2 result matches the rubric, regardless of whether step 1 was correct.
- Judge whether the rubric criterion itself is correct in the rollout, not whether the reasoning that led to it was sound.
- Do not use the rubric or problem text as evidence; only quote from the rollout.

You MUST respond in exactly two sections, in this order:

Section 1 — Reasoning:
1. Criterion: <verbatim criterion including whitespace>.
   Evidence: <verbatim quote from rollout including whitespace> or "not present in rollout".
   Reason: <explain how evidence relates to criterion, maximum 15 words>.
   Verdict: YES/NO.
2. Criterion: ...
   Evidence: ...
   Reason: ...
   Verdict: ...
... and so on

Section 2 — Grades (one line per criterion):
1. YES
2. NO
... and so on

Do NOT output Section 2 before completing Section 1. Do NOT skip Section 1.
"""


def get_rubric_grade_user_prompt(problem, rubric, solution):
    return f"""Problem statement:
{problem}

-------------------

Rubric:
{rubric}

-------------------

Student solution:
{solution}

-------------------

Based on this information, please judge the partial rollout against the rubric with satisfiability as described in the system prompt.
"""


AUDIT_SYSTEM_PROMPT = """You are auditing whether the inputs to a rubric-grading pipeline look structurally well-formed. You will be shown:
1. A problem statement.
2. A model rollout attempting to solve the problem.
3. A grading rubric.

Check ONLY structural well-formedness — do NOT judge correctness, logical consistency, or whether the rollout actually answers correctly:
- The problem statement is plausible (not empty, not raw answer text, not a stray fragment).
- The rollout's first sentence is coherent (not mid-word/mid-sentence) and the rollout is on-topic for the problem. The end may be truncated; that is fine.
- The rubric is a numbered list of criteria that is on-topic for the problem.

Reasoning, factual, and arithmetic errors in the rollout are NOT audit failures — those are exactly what the downstream grader is for.

Respond with a brief explanation (1-3 sentences), then a final line containing exactly `VERDICT: YES` or `VERDICT: NO`.
Answer YES if all three are structurally well-formed, NO otherwise.
"""


def get_audit_user_prompt(problem, rubric, rollout):
    return f"""# Problem statement

{problem}

# Rollout

<rollout>
{rollout}
</rollout>

# Rubric

{rubric}
"""


OPSD_AUDIT_SYSTEM_PROMPT = """You are auditing whether an OPSD training example is structurally well-formed. You will be shown:
1. A problem statement.
2. Additional context that will condition a teacher model.
3. A student rollout produced from the original problem prompt.

Check ONLY this narrow condition: The beginning of the student rollout is coherent and at least loosely related to the problem such that it could plausibly be the start of a response to that problem.

Accept normal solution openings even if they are not complete standalone sentences, including headings, setup fragments, or list-style starts like "We are given:", "Let ...", "Given:", "Solution:", or a first bullet. Only fail if the rollout appears to start mid-word/mid-sentence, is empty, nonsensical, boilerplate unrelated to the problem, or clearly belongs to a different task.

Do NOT judge correctness, logical consistency, arithmetic, final answer quality, or whether the additional context is useful. Misinterpreting the problem is NOT an audit failure if the beginning is a coherent on-topic start. Ignore later parts of the rollout except when needed to understand the opening. The rollout may be truncated at the end; that is fine.

Respond with a brief explanation (1-3 sentences), then a final line containing exactly `VERDICT: YES` or `VERDICT: NO`.
Answer YES if the rollout beginning passes this structural check, NO otherwise.
"""


def get_opsd_audit_user_prompt(problem, opsd_context, student_rollout):
    return f"""# Problem statement

{problem}

# Additional context

<additional_context>
{opsd_context}
</additional_context>

# Student rollout

<student_rollout>
{student_rollout}
</student_rollout>
"""


PROCESS_REWARDS_SYSTEM_PROMPT = """You will be given a math problem along with a solution. They will be formatted as follows:

# Problem
...(math problem)...

# Solution
<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>
...
<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing, verifying, and critiquing the reasoning in detail. Provide the analyses and the conclusion in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>
...
<analysis_n>
...(analysis of paragraph n)...
</analysis_n>
<conclusion>
Correct/Incorrect
</conclusion>

Grading rules:
- When you analyze each paragraph, use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Elaborate on the analysis process carefully.
- If an error is detected in any paragraph, describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach.
- Once a paragraph is found to contain any error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect."
- For instance, given a solution of five paragraphs, if an error is found in the third paragraph, reply with <analysis_1>, <analysis_2>, and <analysis_3> (the latter containing the detailed critique and correction guideline), then <conclusion>Incorrect</conclusion>. Skip the analyses of paragraphs 4 and 5.
- Respond with your analyses and conclusion directly.
"""


PROCESS_REWARDS_AUDIT_SYSTEM_PROMPT = """You are auditing whether the inputs to a process-reward grading pipeline look structurally well-formed. You will be shown:
1. A problem statement.
2. A model rollout split on blank lines and tagged as <paragraph_1>...</paragraph_1>, etc.

Check ONLY structural well-formedness — do NOT judge correctness, logical consistency, arithmetic, or whether the rollout actually answers the problem correctly:
- The problem statement is plausible (not empty, not raw answer text, not a stray fragment).
- Collectively, the paragraphs read as a single response to the problem (on-topic, in a plausible order).

Reasoning, factual, and arithmetic errors in the rollout are NOT audit failures — those are exactly what the downstream grader is for. Misinterpreting the problem is NOT an audit failure if the paragraphs collectively look like a response to the problem.

Respond with a brief explanation (1-2 sentences), then a final line containing exactly `VERDICT: YES` or `VERDICT: NO`.
Answer YES if both conditions are met, NO otherwise.
"""


def get_process_rewards_audit_user_prompt(problem, tagged_rollout):
    return f"""# Problem statement

{problem}

# Rollout

{tagged_rollout}
"""


def get_process_rewards_user_prompt(problem, solution):
    return f"""# Problem
{problem}

# Solution
{solution}

# Task

Provide the analysis/verification/critique/conclusion in the format described above.
"""


def get_state_matching_user_prompt(problem, rubric, solution, is_partial=False):
    rollout_header = '# Partial rollout' if is_partial else '# Rollout'
    rollout_ref = 'partial rollout' if is_partial else 'rollout'
    return f"""# Problem statement

{problem}

{rollout_header}

<rollout>
{solution}
</rollout>

# Rubric

{rubric}

# Instructions

Based on this information, please judge the {rollout_ref} against the rubric with satisfiability as described in the system prompt.
Only the text inside <rollout> ... </rollout> counts as evidence. Ignore the problem statement and rubric when judging satisfaction.
Include the section 1 and section 2 headers.
"""
