"""From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb"""

from dataclasses import dataclass, replace

from lmpo.envs.base import BanditEnv, BaseState

config = {
    'prompt_length': 256,
    'tokens_per_action': 512,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'enable_thinking': 1,
    'format_reward': 0.0,
    'num_tasks': -1,
}


def extract_xml_answer(text: str) -> float:
    try:
        answer = text.split('<answer>')[-1]
        answer = answer.split('</answer>')[0]
        answer = answer.strip().replace(',', '').replace('$', '')
        return float(answer)
    except:
        return -100


def has_formatting(text: str) -> bool:
    return '<answer>' in text and '</answer>' in text


def extract_hash_answer(text: str) -> str | None:
    if '####' not in text:
        raise ValueError("Expected text to contain '####' for answer extraction.")
    return float(text.split('####')[1].strip().replace(',', '').replace('$', ''))


SYSTEM_PROMPT = """
Respond in the following format. Put a single number in the <answer> tag.
<think>
...
</think>
<answer>
...
</answer>"""


@dataclass(frozen=True)
class GSMState(BaseState):
    correct_answer: float = 0.0
    rendered: str = ''


class GSM8KEnv(BanditEnv):
    def __init__(self, tokenizer, train=True, **kwargs):
        merged_kwargs = {**config, **kwargs}
        requested_num_tasks = merged_kwargs.pop('num_tasks', -1)
        super().__init__(**merged_kwargs)
        self.tokenizer = tokenizer
        self.train = train
        if not self.env_nickname:
            base = 'gsm8k' if train else 'gsm8k-test'
            self.env_nickname = f'{base}-{self.seqlen_str()}'
        from datasets import load_dataset

        self.ds = load_dataset('openai/gsm8k', 'main')['train' if train else 'test']
        self.num_tasks = len(self.ds) if requested_num_tasks == -1 else min(requested_num_tasks, len(self.ds))

    def reset(self, idx):
        output_tokens = self.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': self.ds[idx]['question']},
            ],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )
        state = GSMState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            correct_answer=extract_hash_answer(self.ds[idx]['answer']),
        )
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
        action_msg = self.tokenizer.decode(action_tokens)
        reward = 0.0
        evaluated_answer = None
        is_max_tokens = len(action_tokens) == self.tokens_per_action
        if has_formatting(action_msg):
            reward = self.format_reward
            evaluated_answer = extract_xml_answer(action_msg)
            if abs(evaluated_answer - state.correct_answer) < 1e-6:
                reward = 1.0

        render_str = [
            f'{self.tokenizer.decode(state.tokens + action_tokens)}',
            f'Evaluated answer: {evaluated_answer}',
            f'Correct answer: {state.correct_answer}',
            f'Has formatting? {has_formatting(action_msg)}',
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
                'valid_equation': reward > 0.0,
                'correct_answer': reward >= 1.0,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
            },
        )
