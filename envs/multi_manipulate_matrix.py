"""Multi-turn variant of manipulate_matrix: each operation is its own turn."""

from dataclasses import dataclass, field, replace

import numpy as np

from lmpo.envs.base import BaseEnv, BaseState
from lmpo.envs.manipulate_matrix import (
    _clean_desc,
    _generate_example,
    extract_xml_answer,
    has_formatting,
    matrix_to_str,
    normalize_matrix_str,
)


config = {
    'prompt_length': 512,
    'obs_length': 128,
    'tokens_per_action': 1024,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'enable_thinking': 1,
    'matrix_size': 6,
    'n_ops': 5,
    'format_reward': 0.0,
    'max_tokens_penalty': 0.0,
    'num_tasks': -1,
}

SYSTEM_PROMPT = (
    'You will be given a matrix and a sequence of operations. After each operation, '
    'output the resulting matrix in <answer> </answer> tags, one row per line, '
    'values separated by spaces. Example:\n'
    '<answer>\n1 2\n4 5\n</answer>'
)

FIRST_USER_PROMPT = (
    'Initial matrix:\n{matrix}\n\nApply operation 1/{num_turns}: {op}.\n'
    'Output the resulting matrix in <answer> </answer> tags.'
)

NEXT_USER_PROMPT = 'Apply operation {turn}/{num_turns}: {op}.\nOutput the resulting matrix in <answer> </answer> tags.'


@dataclass(frozen=True)
class MultiManipulateMatrixState(BaseState):
    num_turns: int = 0
    op_descriptions: tuple = field(default_factory=tuple)  # length num_turns
    expected_matrix_strs: tuple = field(default_factory=tuple)  # length num_turns; matrix_to_str of steps[t]
    initial_matrix_str: str = ''
    turn_footers: str = ''
    rendered: str = ''
    per_turn_rubrics: tuple = field(default_factory=tuple)  # length num_turns
    per_turn_problems: tuple = field(default_factory=tuple)  # length num_turns
    rubric_n_items: int = 2
    makes_obsolete: str = ''
    turn_rewards: tuple = field(default_factory=tuple)  # length num_turns, filled as turns complete


class MultiManipulateMatrixEnv(BaseEnv):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**{**config, **kwargs})
        self.tokenizer = tokenizer
        self.matrix_size = int(self.matrix_size)
        self.n_ops = int(self.n_ops)
        self.num_turns = self.n_ops
        if not self.env_nickname:
            self.env_nickname = f'multimatrix{self.matrix_size}x{self.n_ops}-{self.seqlen_str()}'

    def get_traj_return(self, per_turn_rewards):
        return per_turn_rewards[:, -1]

    def reset(self, idx):
        row = _generate_example(self.matrix_size, self.n_ops, idx)
        operations = row['metadata'].get('operations', [])
        steps = row['metadata'].get('steps', [])
        initial_matrix = row['metadata'].get('matrix', [])
        op_descs = tuple(_clean_desc(op) for op in operations)
        expected_strs = tuple(matrix_to_str(s) for s in steps)
        initial_matrix_str = matrix_to_str(initial_matrix)
        first_user = FIRST_USER_PROMPT.format(matrix=initial_matrix_str, num_turns=self.num_turns, op=op_descs[0])
        output_tokens = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': first_user}],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )

        per_turn_rubrics = []
        per_turn_problems = []
        for t, expected_str in enumerate(expected_strs):
            rubric = (
                '1. The rollout contains the final answer within <answer> ... </answer> tags.\n\n'
                f'2. The content within the <answer> ... </answer> tags is verbatim:\n{expected_str}'
            )
            ops_so_far = '\n'.join(f'  {i + 1}. {d}' for i, d in enumerate(op_descs[: t + 1]))
            problem = (
                f'Initial matrix:\n{initial_matrix_str}\n\n'
                f'Apply the following operations in order, placing your final answer in <answer> ... </answer> tags.\n{ops_so_far}'
            )
            per_turn_rubrics.append(rubric)
            per_turn_problems.append(problem)

        state = MultiManipulateMatrixState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            turn=0,
            num_turns=self.num_turns,
            op_descriptions=op_descs,
            expected_matrix_strs=expected_strs,
            initial_matrix_str=initial_matrix_str,
            per_turn_rubrics=tuple(per_turn_rubrics),
            per_turn_problems=tuple(per_turn_problems),
        )
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        turn = state.turn
        expected_str = state.expected_matrix_strs[turn]
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
            if hasattr(action_logprobs, 'tolist'):
                action_logprobs = action_logprobs.tolist()
        action_msg = self.tokenizer.decode(action_tokens)

        predicted_str = extract_xml_answer(action_msg)
        reward = 0.0
        is_correct = False
        is_format = has_formatting(action_msg)
        if is_format:
            reward = float(self.format_reward)
            if normalize_matrix_str(predicted_str) == normalize_matrix_str(expected_str):
                reward = 1.0
                is_correct = True
        is_max_tokens = len(action_tokens) >= self.tokens_per_action
        if is_max_tokens:
            reward -= float(self.max_tokens_penalty)

        new_turn = turn + 1
        done = new_turn >= state.num_turns
        if done:
            observation_tokens = []
        else:
            op_desc = state.op_descriptions[new_turn]
            observation_tokens = self.tokenizer.encode('\n') + self.tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': NEXT_USER_PROMPT.format(turn=new_turn + 1, num_turns=self.num_turns, op=op_desc),
                    }
                ],
                add_generation_prompt=True,
                enable_thinking=bool(int(self.enable_thinking)),
            )

        new_turn_rewards = state.turn_rewards + (reward,)
        turn_footer = (
            f'[Turn {turn + 1:2d}/{state.num_turns}]'
            f'  op: {state.op_descriptions[turn]}\n'
            f'    Evaluated answer: {predicted_str!r}\n'
            f'      Correct answer: {expected_str!r}\n'
            f'              Reward: {reward:.2f}'
        )
        env_dense_reward = float(np.mean(new_turn_rewards))
        if done:
            env_return = float(new_turn_rewards[-1])
            turn_footer += f'\nenv_return: {env_return:.2f}  env_dense_reward: {env_dense_reward:.2f}'
        new_footers = (state.turn_footers + '\n' + turn_footer).lstrip('\n')
        transcript = self.tokenizer.decode(state.tokens + action_tokens)
        render_str = transcript + '\n' + new_footers

        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        next_state = replace(
            state,
            tokens=state.tokens + action_tokens + observation_tokens,
            turn_rewards=new_turn_rewards,
            logprobs=new_logprobs + [0] * len(observation_tokens),
            turn=new_turn,
            turn_footers=new_footers,
            rendered=render_str,
        )

        return (
            next_state,
            observation_tokens,
            reward,
            done,
            {
                'has_formatting': is_format,
                'correct_answer': is_correct,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
                'env_dense_reward': env_dense_reward,
            },
        )
