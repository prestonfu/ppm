"""Helpers for filtering HuggingFace math/reasoning datasets."""

import re

_LETTER = r'[A-E]'
_OPTION_PATTERNS = [
    rf'(?:^|\n)\s*\\textbf\{{\(?({_LETTER})\)?\}}',
    rf'(?:^|\n)\s*\(({_LETTER})\)\s',
    rf'(?:^|\n)\s*({_LETTER})\)\s',
    rf'(?:^|\n)\s*({_LETTER})\.\s',
    rf'\\textbf\{{\(?({_LETTER})\)?\}}\\?\s',
    rf'\\text\{{\s*\(?({_LETTER})\)?\s*\}}',
    rf'\\mathrm\{{\s*\(?({_LETTER})\)?\s*\}}',
    rf'\$\(?({_LETTER})\)?\$\s*[\.\)]\s',
]
_OPTION_RES = [re.compile(p) for p in _OPTION_PATTERNS]
_TYPE_LETTER_RE = re.compile(r'type the letter', re.IGNORECASE)


def is_multiple_choice(problem: str, min_letters: int = 3) -> bool:
    """Heuristic: True if ``problem`` looks like A-E multiple choice.

    Flags when >=``min_letters`` distinct uppercase letters appear as option
    markers in any common style (``\\textbf{(A)}``, ``A)``, ``A.``, ``(A)``,
    ``\\text{(A)}``, ``\\mathrm{(A)}``, ``$A$.``), or when the prompt asks the
    reader to "type the letter".

    Lowercase is excluded so that "(a) ... (b) ..." sub-problem labels don't
    trigger the filter.
    """
    if not problem:
        return False
    if _TYPE_LETTER_RE.search(problem):
        return True
    seen = set()
    for r in _OPTION_RES:
        for m in r.findall(problem):
            seen.add(m)
            if len(seen) >= min_letters:
                return True
    return False


def filter_multiple_choice(ds, problem_field: str = 'problem'):
    """Drop multiple-choice rows from a HuggingFace ``Dataset``."""
    return ds.filter(lambda row: not is_multiple_choice(row[problem_field]))
