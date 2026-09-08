from dataclasses import dataclass

from lmpo.envs.base_math import BaseRubricMathEnv, MathWithOracleState, base_math_config


config = {
    **base_math_config,
    'n_ops': 8,
    'use_obsoletion_rubric': 0,
    'prefix_paragraphs': 0,
    'prefix_fraction': 0.0,
}


@dataclass(frozen=True)
class GsmInfiniteHardState(MathWithOracleState):
    prefix_paragraphs_used: int = 0


class GsmInfiniteHardEnv(BaseRubricMathEnv):
    state_cls = GsmInfiniteHardState
    slug = 'gsm_infinite_hard'

    def __init__(
        self,
        tokenizer,
        n_ops=8,
        use_obsoletion_rubric=0,
        prefix_paragraphs=0,
        prefix_fraction=0.0,
        env_nickname='',
        **kwargs,
    ):
        self.n_ops = int(n_ops)
        self.use_obsoletion_rubric = bool(use_obsoletion_rubric)
        self.prefix_paragraphs = int(prefix_paragraphs)
        self.prefix_fraction = float(prefix_fraction)
        assert not (self.prefix_paragraphs > 0 and self.prefix_fraction > 0), (
            'Cannot use both prefix_paragraphs and prefix_fraction'
        )
        assert 0.0 <= self.prefix_fraction <= 1.0, 'prefix_fraction must be in [0, 1]'
        super().__init__(tokenizer, env_nickname=env_nickname, **kwargs)
        self.slug += '_boxed'
        if not env_nickname:
            self.env_nickname = f'{self.slug}-ops{self.n_ops}-{self.seqlen_str()}'

    def load_dataset(self):
        from datasets import load_dataset

        self.ds = load_dataset(f'prestonfu/gsm_infinite_hard_r0.4_ops{self.n_ops}')['train_boxed']
        if self.shuffle:
            self.ds = self.ds.shuffle(seed=42)
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx):
        return str(self.ds[idx]['answer'])

    def chat_messages(self, idx):
        row = self.ds[idx]
        return [
            {'role': 'system', 'content': row['system_prompt']},
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

    def reset(self, idx):
        row = self.ds[idx]
        rubric_lines = [line for line in row['rubric'].split('\n') if line.strip()]
        prefix_paragraphs_used = 0
        if self.prefix_paragraphs > 0 or self.prefix_fraction > 0:
            output_tokens, prefix_paragraphs_used = self.tokens_with_prefix(
                idx, row['synthetic_solution'], self.prefix_paragraphs, self.prefix_fraction
            )
        else:
            output_tokens = self.tokenize_chat(self.chat_messages(idx))
        makes_obsolete = ''
        if self.use_obsoletion_rubric:
            v = row.get('makes_obsolete', '')
            makes_obsolete = '' if v is None else str(v)
        state = self.state_cls(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            correct_answer=self.correct_answer(idx),
            oracle_solution=row['synthetic_solution'],
            opsd_context=row['synthetic_solution'],
            rubric=row['rubric'],
            rubric_n_items=len(rubric_lines),
            problem=row['problem'],
            prefix_paragraphs_used=prefix_paragraphs_used,
            makes_obsolete=makes_obsolete,
        )
        self.check_rubric_format(state)
        return state, output_tokens
