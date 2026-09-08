"""Rubric-generation prompts used by dataset/rubric builder scripts."""

from lmpo.scripts.rubrics.rubric_examples import (
    OBSOLETION_EXAMPLES,
    SAMPLE_PROBLEM,
    SAMPLE_REFERENCE_SOLUTION,
    SAMPLE_RUBRIC,
)

OBSOLETION_SPARSE = """\
For the obsoletion graph, include only dependencies where satisfying one item makes another logically redundant for grading. A later item may obsolete an earlier item only if it explicitly contains the earlier mathematical content or could not reasonably be satisfied without it. Do not let the final answer obsolete the reasoning leading to it. Prefer a sparse graph.\
"""

OBSOLETION_MEDIUM = """\
For the obsoletion graph, a later item may obsolete earlier steps when it clearly depends on them or substantially subsumes their mathematical content. Allow natural local chains, such as a derived inequality obsoleting the intermediate comparison that directly produced it, or a case resolution obsoleting merely considering that case. Do not let a final answer obsolete specific derivations, constructions, case analyses, or impossibility arguments. Prefer a reasonably sparse graph that reduces redundant grading without skipping substantive reasoning.\
"""

OBSOLETION_DENSE = """\
For the obsoletion graph, later items may obsolete earlier setup, intermediate steps, or prerequisite observations when those earlier items are normally required to obtain the later result in the reference solution. Allow multi-item dependencies when a combination of later items jointly subsumes earlier reasoning. The final answer may obsolete only items that merely restate the same conclusion or broad bound, not items requiring specific definitions, derivations, constructions, case analyses, counting arguments, or impossibility arguments. Prefer grading compression, but avoid edges based only on chronology, topic similarity, or weak correlation.\
"""

OBSOLETION_BLOCKS = {
    'sparse': OBSOLETION_SPARSE,
    'medium': OBSOLETION_MEDIUM,
    'dense': OBSOLETION_DENSE,
}


CREATE_RUBRIC_SYSTEM_PROMPT = f"""\
You are given a problem statement and a reference solution that should be treated as correct. Generate a rubric for evaluating other solutions against the reasoning actually present in the reference solution.

The rubric must be faithful to the reference solution. Do not add lemmas, cases, rigor, bounds, or proof obligations that are not explicitly present or clearly implied. If the reference solution uses an approximation or informal argument, describe that argument as it appears rather than replacing it with a stronger rigorous version. Rubric items should test whether another solution follows the same essential reasoning path as the reference solution.

Each rubric item must:
- be a single sentence;
- be evaluable as YES or NO without extra context;
- state one concrete mathematical requirement, such as deriving a claim, comparing expressions, handling a case, or reaching a conclusion;
- represent one logically distinct requirement, with no duplicates;
- be explicitly stated or clearly implied by the reference solution.

The final rubric item must state the final conclusion or answer.

Do not include:
- definitions, formulas, or background facts unless essential to the solution’s reasoning;
- vague criteria such as "correctly analyzes," "uses appropriate reasoning," or "understands the problem";
- presentation or style criteria;
- point values, grading guidance, solution text, hints, or references to these instructions;
- stronger arguments than the reference solution gives;
- requirements needed only for a fully rigorous proof but absent from the reference solution.

## Output format

Short breakdown of the reference solution. (4 sentences max)

Estimated number of rubric items for each key step. (2 sentences max)

<rubric>
1. Rubric item 1
2. Rubric item 2
and so on... (6-12 items)
</rubric>

## Example problem statement
{SAMPLE_PROBLEM}

## Example reference solution
{SAMPLE_REFERENCE_SOLUTION}

## Example rubric
{SAMPLE_RUBRIC}
"""


CREATE_RUBRIC_USER_PROMPT = """\
## Problem statement
{problem}

## Reference solution
{solution}

## Instructions
Provide the solution breakdown, rubric item allocation, and rubric as described in the system prompt.\
"""


def obsoletionsystem_prompt(obsoletion_mode='medium'):
    assert obsoletion_mode in OBSOLETION_BLOCKS, (
        f'obsoletion_mode must be one of {sorted(OBSOLETION_BLOCKS)}, got {obsoletion_mode!r}'
    )
    return f"""\
You are given a problem statement and a grading rubric (a numbered list of criteria). Construct an obsoletion graph: a set of rules stating that satisfying some rubric items makes other items redundant to grade, because the satisfied items already imply or subsume them.

{OBSOLETION_BLOCKS[obsoletion_mode]}

Each rule has the form <rule if_all="i,j" makes_obsolete="a,b,c" />, meaning that if items i and j are all satisfied, then items a, b, and c need not be graded separately. Only reference item numbers that exist in the rubric.

## Output format

Brief explanation of the dependency policy used in the obsoletion graph. (4 sentences max)

<makes_obsolete_graph>
<rule if_all="..." makes_obsolete="..." />
<rule if_all="..." makes_obsolete="..." />
and so on...
</makes_obsolete_graph>

## Example problem statement
{SAMPLE_PROBLEM}

## Example rubric
{SAMPLE_RUBRIC}

## Example obsoletion graph
{OBSOLETION_EXAMPLES[obsoletion_mode]}
"""


OBSOLETION_USER_PROMPT = """\
## Problem statement
{problem}

## Rubric
{rubric}

## Instructions
Provide the dependency policy explanation and the obsoletion graph as described in the system prompt.\
"""
