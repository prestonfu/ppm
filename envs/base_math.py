"""Single-turn math / proof envs with boxed final-answer formatting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace

from lmpo.envs.base import BanditEnv, BaseState
from lmpo.utils.math_grading import grade_answer

# Default hyperparameters for deepscaler-like envs; subclasses may override ``config`` dicts.
base_math_config = {
    'prompt_length': 256,
    'tokens_per_action': 4096,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'format_reward': 0.0,
    'enable_thinking': 1,
    'num_tasks': -1,
    'offset': 0,
    'subset': '',
    'shuffle': 1,
}

BASE_MATH_SYSTEM_PROMPT = (
    r'Write your final answer in LaTeX as \boxed{...}. '
    'Ignore all other answer formatting instructions.'
)


_BOXED_START_RE = re.compile(r'\\boxed\{')


def extract_xml_answer_boxed(text: str) -> str:
    """Extract content of last \\boxed{...}, handling nested braces."""
    matches = list(_BOXED_START_RE.finditer(text))
    if not matches:
        return ''
    m = matches[-1]
    depth = 1
    i = m.end()
    while i < len(text) and depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    content = text[m.end() : i - 1] if depth == 0 else ''
    # Leave commas and dollar signs for grade_answer to normalize.
    return content.strip()


def has_formatting_boxed(text: str) -> bool:
    return bool(_BOXED_START_RE.search(text))


@dataclass(frozen=True)
class BaseMathState(BaseState):
    correct_answer: str = ''
    rendered: str = ''


@dataclass(frozen=True)
class MathWithOracleState(BaseMathState):
    oracle_solution: str = ''
    opsd_context: str = ''
    rubric: str = ''
    rubric_n_items: int = -1
    makes_obsolete: str = ''
    problem: str = ''


class BaseMathEnv(BanditEnv):
    state_cls: type = BaseMathState
    slug: str = 'base_math'

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        merged_kwargs = {**base_math_config, **kwargs}
        requested_num_tasks = merged_kwargs.pop('num_tasks', -1)
        offset = int(merged_kwargs.pop('offset', 0))
        subset = merged_kwargs.pop('subset', '')
        self.shuffle = int(merged_kwargs.pop('shuffle', 1))
        super().__init__(**merged_kwargs)
        if not self.env_nickname:
            self.env_nickname = f'{self.slug}-{self.seqlen_str()}'
        self.load_dataset()
        if subset:
            subset_path = subset if os.path.isabs(subset) else os.path.join(os.path.dirname(__file__), subset)
            with open(subset_path) as f:
                subset_idxs = [int(line.strip()) for line in f if line.strip()]
            self.ds = self.ds.select(subset_idxs)
            self.num_tasks = len(self.ds)
        if requested_num_tasks != -1:
            self.num_tasks = min(requested_num_tasks, self.num_tasks) if self.num_tasks != -1 else requested_num_tasks
        if offset:
            assert self.num_tasks != -1, 'offset requires a known num_tasks'
            assert offset + self.num_tasks <= len(self.ds), (
                f'offset={offset} + num_tasks={self.num_tasks} exceeds dataset size {len(self.ds)}'
            )
            self.ds = self.ds.select(range(offset, offset + self.num_tasks))

    def load_dataset(self):
        raise NotImplementedError

    def correct_answer(self, idx):
        raise NotImplementedError

    def chat_messages(self, idx):
        raise NotImplementedError

    def system_prompt(self):
        return BASE_MATH_SYSTEM_PROMPT

    def check_rubric_format(self, state):
        pass

    def reset(self, idx):
        output_tokens = self.tokenizer.apply_chat_template(
            self.chat_messages(idx),
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )
        state = self.state_cls(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            correct_answer=self.correct_answer(idx),
        )
        self.check_rubric_format(state)
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens, action_logprobs=None):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
        action_msg = self.tokenizer.decode(action_tokens)
        reward = 0.0
        evaluated_answer = None
        is_max_tokens = len(action_tokens) == self.tokens_per_action
        has_format = has_formatting_boxed(action_msg)
        if has_format:
            reward = self.format_reward
            evaluated_answer = extract_xml_answer_boxed(action_msg)
            if grade_answer(evaluated_answer, state.correct_answer):
                reward = 1.0

        render_str = [
            f'{self.tokenizer.decode(state.tokens + action_tokens)}',
            f'Evaluated answer: {evaluated_answer}',
            f'Correct answer: {state.correct_answer}',
            f'Has formatting? {has_format}',
            f'Reward: {reward:.2f}',
        ]
        render_str = '\n'.join(render_str)
        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        state = replace(state, tokens=state.tokens + action_tokens, logprobs=new_logprobs, rendered=render_str)
        return (
            state,
            [],
            reward,
            True,
            {
                'has_formatting': has_format,
                'correct_answer': reward >= 1.0,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
            },
        )

    def get_sft_data(self, idx):
        state, prompt_tokens = self.reset(idx)
        oracle_solution = getattr(state, 'oracle_solution', '')
        assert oracle_solution, f'{type(self).__name__} state does not provide oracle_solution'
        target_tokens = self.tokenizer.encode(oracle_solution)
        eos_id = self.tokenizer.eos_token_id
        if not target_tokens or target_tokens[-1] != eos_id:
            target_tokens = list(target_tokens) + [eos_id]
        prompt_tokens = list(prompt_tokens)
        target_tokens = list(target_tokens)
        tokens = prompt_tokens + target_tokens
        target_mask = [0] * len(prompt_tokens) + [1] * len(target_tokens)
        return tokens, target_mask


class BaseRubricMathEnv(BaseMathEnv):
    def format_instruction(self) -> str:
        return r'**Important**: Show your reasoning step-by-step, and present the final answer as \boxed{...}.'

    def final_instruction(self) -> str:
        return r'Make sure your final answer is written as: \boxed{your answer here}'

    def tokenize_chat(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )

    def check_rubric_format(self, state):
        expected = r'\boxed{'
        if expected not in state.rubric:
            raise ValueError(f'rubric missing {expected!r}:\n{state.rubric}')

    def tokens_with_prefix(self, idx, solution, n_paragraphs=0, fraction=0.0):
        n_paragraphs = int(n_paragraphs)
        fraction = float(fraction)
        if n_paragraphs <= 0 and fraction <= 0:
            return self.tokenize_chat(self.chat_messages(idx)), 0
        assert not (n_paragraphs > 0 and fraction > 0), 'Cannot use both n_paragraphs and fraction'
        assert 0.0 <= fraction <= 1.0, 'fraction must be in [0, 1]'

        paragraphs = str(solution).strip().split('\n\n')
        target_n = max(1, int(len(paragraphs) * fraction)) if fraction > 0 else n_paragraphs
        for n in range(min(target_n, len(paragraphs)), 0, -1):
            partial = '\n\n'.join(paragraphs[:n])
            output_tokens = self.tokenize_chat(
                self.chat_messages_with_prefix(idx, partial, is_full=n == len(paragraphs))
            )
            if len(output_tokens) <= self.prompt_length:
                return output_tokens, n
        return self.tokenize_chat(self.chat_messages(idx)), 0
