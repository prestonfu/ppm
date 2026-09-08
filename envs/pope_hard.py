import re
from dataclasses import dataclass

from lmpo.envs.base_math import (
    BASE_MATH_SYSTEM_PROMPT,
    BaseRubricMathEnv,
    MathWithOracleState,
    BaseMathEnv,
    BaseMathState,
    base_math_config,
)

config = {
    **{key: value for key, value in base_math_config.items() if key != 'subset'},
    'split': 'train',
    'prefix_paragraphs': 0,
    'prefix_fraction': 0.0,
    'rubric_source': 'gemini-3-flash-preview',
    'use_obsoletion_rubric': 0,
}


_ANSWER_HEADING_RE = re.compile(r'^(?:\*\*)?\s*(?:final\s+)?answer\s*(?:\*\*)?\s*:', flags=re.IGNORECASE)
_TRAILING_XML_ANSWER_RE = re.compile(r'(?:\n\s*)*<answer>.*?</answer>\s*\Z', flags=re.IGNORECASE | re.DOTALL)


def _preprocess_gemini_solution(row):
    solution = str(row['gemini_solution']).strip()
    answer = str(row['answer']).strip()
    solution = _TRAILING_XML_ANSWER_RE.sub('', solution).rstrip()
    paragraphs = re.split(r'\n\s*\n', solution)
    if paragraphs and _ANSWER_HEADING_RE.match(paragraphs[-1].strip()):
        solution = '\n\n'.join(paragraphs[:-1]).rstrip()
    else:
        lines = solution.splitlines()
        if lines and _ANSWER_HEADING_RE.match(lines[-1].strip()):
            solution = '\n'.join(lines[:-1]).rstrip()
    return f'{solution}\n\n<answer> {answer} </answer>' if solution else f'<answer> {answer} </answer>'


@dataclass(frozen=True)
class PopeHardState(MathWithOracleState):
    prefix_paragraphs_used: int = 0


class PopeHardEnv(BaseRubricMathEnv):
    state_cls = PopeHardState
    slug = 'pope_hard'

    def __init__(
        self,
        tokenizer,
        split='train',
        prefix_paragraphs=0,
        prefix_fraction=0.0,
        rubric_source='',
        use_obsoletion_rubric=0,
        **kwargs,
    ):
        assert rubric_source in ['Qwen3-4B-Instruct-2507', 'gemini-3-flash-preview']
        self.split = split
        self.rubric_source = rubric_source
        self.use_obsoletion_rubric = bool(use_obsoletion_rubric)
        super().__init__(tokenizer, **kwargs)
        self.prefix_paragraphs = int(prefix_paragraphs)
        self.prefix_fraction = float(prefix_fraction)
        assert not (self.prefix_paragraphs > 0 and self.prefix_fraction > 0), (
            'Cannot use both prefix_paragraphs and prefix_fraction'
        )
        assert 0.0 <= self.prefix_fraction <= 1.0, 'prefix_fraction must be in [0, 1)'

    def format_instruction(self) -> str:
        return '**Important**: Show your reasoning step-by-step, and present the final answer in <answer> ... </answer> tags.'

    def final_instruction(self) -> str:
        return 'Make sure your final answer is written as: <answer> your answer here </answer>'

    def check_rubric_format(self, state):
        expected = '<answer>'
        if expected not in state.rubric:
            raise ValueError(f'rubric missing {expected!r}:\n{state.rubric}')

    def load_dataset(self) -> None:
        from datasets import load_dataset

        assert self.rubric_source == 'gemini-3-flash-preview', (
            f'kvfrans/POPE-HARD-w-oracle-solution-gemini-rubric requires rubric_source=gemini-3-flash-preview, '
            f'got {self.rubric_source!r}'
        )
        self.ds = load_dataset(
            'kvfrans/POPE-HARD-w-oracle-solution-gemini-rubric',
            split=self.split,
        )
        if self.shuffle:
            self.ds = self.ds.shuffle(seed=42)
        self.num_tasks = len(self.ds)
        if self.use_obsoletion_rubric and 'makes_obsolete' not in self.ds.column_names:
            raise ValueError('Dataset missing required "makes_obsolete" column')

    def correct_answer(self, idx) -> str:
        return self.ds[idx]['answer']

    def chat_messages(self, idx) -> list[dict]:
        row = self.ds[idx]
        return [
            {'role': 'system', 'content': BASE_MATH_SYSTEM_PROMPT},
            {'role': 'user', 'content': row['problem']},
        ]

    def chat_messages_with_prefix(self, idx, partial: str, is_full: bool = False) -> list[dict]:
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
        if not self.use_obsoletion_rubric:
            return ''
        v = row.get('makes_obsolete', '')
        return '' if v is None else str(v)

    def reset(self, idx):
        row = self.ds[idx]
        gemini_solution = _preprocess_gemini_solution(row)
        rubric = row['rubric']
        rubric_lines = [l for l in rubric.split('\n') if l.strip()]
        rubric_lines.append(
            f'{len(rubric_lines) + 1}. The answer is correctly formatted within <answer> ... </answer> tags.'
        )
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
