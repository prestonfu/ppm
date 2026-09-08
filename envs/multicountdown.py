from dataclasses import dataclass, field, replace
import numpy as np

from lmpo.envs.base import BaseEnv, BaseState
from lmpo.envs.countdown import generate_expression, score_countdown_action


config = {
    'prompt_length': 512,
    'obs_length': 128,
    'tokens_per_action': 4096,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'enable_thinking': 1,
    'num_terms': 5,
    'num_turns': 2,
    'format_reward': 0.0,
    'max_tokens_penalty': 0.0,
    'num_tasks': -1,
}

SYSTEM_PROMPT = (
    'You are a helpful assistant. You first think about the reasoning process in the mind, and then provide the '
    'user with the answer. You will be given a set of base numbers along with a target number. Your job is to '
    'create an equation that equals the target using basic arithmetic operations (+, -, *, /). Each provided '
    'number must be used exactly once. Show your work in <think> </think> tags. Think for only ten sentences, '
    'then return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. '
    'You will solve countdown problems multiple times, and each problem will be a different set of numbers and '
    'target.'
)
USER_PROMPT = '[Problem {turn}/{num_turns}] Base Numbers: {numbers}. Target: {target}.'


@dataclass(frozen=True)
class MultiCountdownState(BaseState):
    num_turns: int = 0
    num_terms: int = 0
    task_numbers_flat: list = field(default_factory=list)  # length num_turns * num_terms
    task_targets: list = field(default_factory=list)  # length num_turns
    turn_footers: str = ''  # newline-joined footer per completed turn
    rendered: str = ''

    def numbers_for_turn(self, turn):
        s = turn * self.num_terms
        return list(self.task_numbers_flat[s : s + self.num_terms])


class MultiCountdownEnv(BaseEnv):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**{**config, **kwargs})
        self.tokenizer = tokenizer
        self.num_terms = int(self.num_terms)
        self.num_turns = int(self.num_turns)
        if not self.env_nickname:
            self.env_nickname = f'multicountdown-{self.num_terms}x{self.num_turns}'

    def get_traj_return(self, per_turn_rewards):
        return np.all(per_turn_rewards >= 1, axis=1).astype(np.float32)

    def generate_all_tasks(self, idx):
        """All num_turns tasks fully determined by idx — no per-rollout RNG drift."""
        rng = np.random.RandomState(idx)
        numbers_flat, targets = [], []
        for _ in range(self.num_turns):
            _, numbers, target = generate_expression(rng, self.num_terms)
            rng.shuffle(numbers)
            assert len(numbers) == self.num_terms
            numbers_flat.extend(int(n) for n in numbers)
            targets.append(int(target))
        return numbers_flat, targets

    def user_prompt(self, turn, numbers, target):
        return USER_PROMPT.format(turn=turn + 1, num_turns=self.num_turns, numbers=list(numbers), target=target)

    def make_turn_observation(self, turn, numbers, target):
        return self.tokenizer.encode('\n') + self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': self.user_prompt(turn, numbers, target)}],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )

    def reset(self, idx):
        numbers_flat, targets = self.generate_all_tasks(idx)
        first_numbers = numbers_flat[: self.num_terms]
        first_target = targets[0]
        output_tokens = self.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': self.user_prompt(0, first_numbers, first_target)},
            ],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )
        state = MultiCountdownState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            turn=0,
            num_turns=self.num_turns,
            num_terms=self.num_terms,
            task_numbers_flat=numbers_flat,
            task_targets=targets,
        )
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        turn = state.turn
        current_numbers = state.numbers_for_turn(turn)
        current_target = state.task_targets[turn]
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
            if hasattr(action_logprobs, 'tolist'):
                action_logprobs = action_logprobs.tolist()
        action_msg = self.tokenizer.decode(action_tokens)

        reward, evaluated_answer, is_valid = score_countdown_action(
            current_numbers, current_target, action_msg, self.format_reward
        )
        is_max_tokens = len(action_tokens) >= self.tokens_per_action
        if is_max_tokens:
            reward -= float(self.max_tokens_penalty)

        new_turn = turn + 1
        done = new_turn >= state.num_turns
        if done:
            observation_tokens = []
        else:
            observation_tokens = self.make_turn_observation(
                new_turn, state.numbers_for_turn(new_turn), state.task_targets[new_turn]
            )

        turn_footer = (
            f'[Turn {turn + 1}/{state.num_turns}] '
            f'numbers={current_numbers} target={current_target} '
            f'evaluated={evaluated_answer} valid={is_valid} reward={reward:.2f}'
        )
        new_footers = (state.turn_footers + '\n' + turn_footer).lstrip('\n')
        transcript = self.tokenizer.decode(state.tokens + action_tokens)
        render_str = transcript + '\n' + new_footers
        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        next_state = replace(
            state,
            tokens=state.tokens + action_tokens + observation_tokens,
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
                'valid_equation': is_valid,
                'correct_answer': reward >= 1.0,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
            },
        )
