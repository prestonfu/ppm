from dataclasses import dataclass, replace

from lmpo.envs.base import BanditEnv, BaseState

config = {
    'prompt_length': 256,
    'tokens_per_action': 128,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'enable_thinking': 0,
    'num_tasks': -1,
}

POEM_TOPICS = (
    'cat',
    'dog',
    'bird',
    'fish',
    'elephant',
    'tiger',
    'lion',
    'giraffe',
    'zebra',
    'monkey',
)


@dataclass(frozen=True)
class PoemState(BaseState):
    pass


class PoemLengthEnv(BanditEnv):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**{**config, **kwargs})
        self.tokenizer = tokenizer
        if not self.env_nickname:
            self.env_nickname = 'poem'

    def reset(self, idx):
        msg = f'Write three sentences about {POEM_TOPICS[idx % len(POEM_TOPICS)]}'
        output_tokens = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': msg}],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )
        state = PoemState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
        )
        return state, output_tokens

    def render(self, state):
        return self.tokenizer.decode(state.tokens)

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
        action_msg = self.tokenizer.decode(action_tokens)
        reward = len(action_msg)
        is_max_tokens = len(action_tokens) >= self.tokens_per_action
        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        state = replace(state, tokens=state.tokens + action_tokens, logprobs=new_logprobs)
        return state, [], reward, True, {'is_max_tokens': is_max_tokens}
