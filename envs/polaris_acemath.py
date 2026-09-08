import os
import re
from dataclasses import dataclass

from lmpo.envs.base_math import BaseRubricMathEnv, MathWithOracleState, base_math_config
from lmpo.utils.rubric_utils import union_makes_obsolete


config = {
    **base_math_config,
    'dataset_name': 'prestonfu/polaris-acemath-gemini-rubrics-v2',
    'split': 'train',
    'prefix_paragraphs': 0,
    'prefix_fraction': 0.0,
    'rubric_source': 'gemini-3-flash-preview',
    'use_obsoletion_rubric': '',
}


OBSOLETION_LEVELS = {
    '': [],
    'sparse': ['sparse'],
    'medium': ['sparse', 'medium'],
    'dense': ['sparse', 'medium', 'dense'],
}

_TRAILING_XML_ANSWER_RE = re.compile(r'(?:\n\s*)*<answer>.*?</answer>\s*\Z', flags=re.IGNORECASE | re.DOTALL)
_TRAILING_BOXED_ANSWER_RE = re.compile(r'(?:\n\s*)*\\boxed\{.*\}\s*\Z', flags=re.DOTALL)
_BOXED_START_RE = re.compile(r'\\boxed\{')


def _has_trailing_boxed(text):
    matches = list(_BOXED_START_RE.finditer(text))
    if not matches:
        return False
    m = matches[-1]
    depth = 1
    i = m.end()
    while i < len(text) and depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return depth == 0 and not text[i:].strip()


def _boxed(answer):
    return r'\boxed{' + str(answer).strip() + '}'


def _preprocess_gemini_solution(row):
    solution = str(row.get('gemini_solution', '')).strip()
    answer = str(row['answer']).strip()
    if not solution:
        return _boxed(answer)
    if _has_trailing_boxed(solution):
        return solution
    solution = _TRAILING_XML_ANSWER_RE.sub('', solution).rstrip()
    solution = _TRAILING_BOXED_ANSWER_RE.sub('', solution).rstrip()
    return f'{solution}\n\n{_boxed(answer)}'


def _normalize_obsoletion_level(value):
    if value in (False, None, 0, '0'):
        return ''
    if value in (True, 1, '1'):
        return 'dense'
    return str(value)


@dataclass(frozen=True)
class PolarisAcemathState(MathWithOracleState):
    prefix_paragraphs_used: int = 0


class PolarisAcemathEnv(BaseRubricMathEnv):
    state_cls = PolarisAcemathState
    slug = 'polaris_acemath'

    def __init__(
        self,
        tokenizer,
        dataset_name='prestonfu/polaris-acemath-gemini-rubrics-v2',
        split='train',
        prefix_paragraphs=0,
        prefix_fraction=0.0,
        rubric_source='gemini-3-flash-preview',
        use_obsoletion_rubric='',
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.split = 'train_boxed' if split == 'train' else split
        self.rubric_source = rubric_source
        self.use_obsoletion_rubric = _normalize_obsoletion_level(use_obsoletion_rubric)
        if self.use_obsoletion_rubric not in OBSOLETION_LEVELS:
            raise ValueError(
                f'use_obsoletion_rubric must be one of {sorted(OBSOLETION_LEVELS)}, got {use_obsoletion_rubric!r}'
            )
        super().__init__(tokenizer, **kwargs)
        self.prefix_paragraphs = int(prefix_paragraphs)
        self.prefix_fraction = float(prefix_fraction)
        assert not (self.prefix_paragraphs > 0 and self.prefix_fraction > 0), (
            'Cannot use both prefix_paragraphs and prefix_fraction'
        )
        assert 0.0 <= self.prefix_fraction <= 1.0, 'prefix_fraction must be in [0, 1]'

    def load_dataset(self):
        from datasets import DatasetDict, load_dataset, load_from_disk

        if os.path.exists(self.dataset_name):
            ds = load_from_disk(self.dataset_name)
            if isinstance(ds, DatasetDict):
                ds = ds[self.split]
        else:
            ds = load_dataset(self.dataset_name)[self.split]
        required_cols = {'problem', 'answer', 'gemini_solution', 'rubric', 'rubric_model'}
        missing = required_cols - set(ds.column_names)
        if missing:
            raise ValueError(f'{self.dataset_name!r} missing required columns: {sorted(missing)}')
        for level in OBSOLETION_LEVELS[self.use_obsoletion_rubric]:
            col = f'makes_obsolete_{level}'
            if col not in ds.column_names:
                raise ValueError(f'{self.dataset_name!r} missing required column: {col!r}')
        ds = ds.filter(lambda row: bool(str(row['gemini_solution']).strip()) and bool(str(row['rubric']).strip()))
        if self.rubric_source:
            bad = set(ds['rubric_model']) - {self.rubric_source}
            if bad:
                raise ValueError(f'rubric_model column must be {self.rubric_source!r}; got extras {sorted(bad)}')
        self.ds = ds.shuffle(seed=42) if self.shuffle else ds
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx):
        return str(self.ds[idx]['answer'])

    def chat_messages(self, idx):
        row = self.ds[idx]
        return [
            {'role': 'system', 'content': self.system_prompt()},
            {'role': 'user', 'content': row['problem']},
        ]

    def chat_messages_with_prefix(self, idx, partial: str, is_full: bool = False):
        row = self.ds[idx]
        label = 'reference solution' if is_full else 'partial solution'
        header = 'Reference Solution' if is_full else 'Partial Solution'
        content = (
            f'You are given a problem and a {label}. Your task is to carefully study the {label} '
            f'and use it as guidance to derive a complete and correct solution.\n\n'
            f'Use the information from the {label} silently. Do not copy, rephrase, or explicitly mention anything from it.\n\n'
            f'{self.format_instruction()}\n\n'
            f'# Problem\n{row["problem"]}\n\n'
            f'# {header}\n{partial}\n\n'
            'Derive the full solution to the problem, re-deriving each step yourself. '
            f'{self.final_instruction()}'
        )
        return [{'role': 'user', 'content': content}]

    def get_makes_obsolete(self, row):
        levels = OBSOLETION_LEVELS[self.use_obsoletion_rubric]
        if not levels:
            return ''
        return union_makes_obsolete([row[f'makes_obsolete_{level}'] for level in levels])

    def reset(self, idx):
        row = self.ds[idx]
        gemini_solution = _preprocess_gemini_solution(row)
        rubric_lines = [line for line in str(row['rubric']).split('\n') if line.strip()]
        rubric_lines.append(rf'{len(rubric_lines) + 1}. The final answer is written in \boxed{{}} format.')
        rubric = '\n'.join(rubric_lines)
        prefix_paragraphs_used = 0

        if self.prefix_paragraphs > 0 or self.prefix_fraction > 0:
            output_tokens, prefix_paragraphs_used = self.tokens_with_prefix(
                idx, gemini_solution, self.prefix_paragraphs, self.prefix_fraction
            )
        else:
            output_tokens = self.tokenize_chat(self.chat_messages(idx))

        state = self.state_cls(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            correct_answer=self.correct_answer(idx),
            oracle_solution=gemini_solution,
            opsd_context=gemini_solution,
            rubric=rubric,
            rubric_n_items=len(rubric_lines),
            prefix_paragraphs_used=prefix_paragraphs_used,
            makes_obsolete=self.get_makes_obsolete(row),
            problem=row['problem'],
        )
        self.check_rubric_format(state)
        return state, output_tokens
