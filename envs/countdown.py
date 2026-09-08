"""From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/games/countdown.py"""

from dataclasses import dataclass, field, replace
import numpy as np
import re
from sympy import symbols

from lmpo.envs.base import BanditEnv, BaseState

config = {
    'prompt_length': 256,
    'tokens_per_action': 1024,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'enable_thinking': 1,
    'num_terms': 4,
    'format_reward': 0.0,
    'max_tokens_penalty': 0.0,
    'num_tasks': -1,
}

SYSTEM_PROMPT = 'You are a helpful assistant. You first think about the reasoning process in the mind, and then provide the user with the answer.'
USER_PROMPT = 'Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. Think for only ten sentences, then return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.'


@dataclass(frozen=True)
class CountdownState(BaseState):
    numbers: list = field(default_factory=list)
    correct_answer: int = 0
    rendered: str = ''


def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split('<answer>')[-1]
        answer = answer.split('</answer>')[0]
        return answer
    except:
        return 'Parsing error'


def valid_equation_for_numbers(numbers, text: str) -> bool:
    try:
        equation_str = extract_xml_answer(text)
        numbers_in_eq = sorted(int(n) for n in re.findall(r'\d+', equation_str))
        return numbers_in_eq == sorted(numbers)
    except:
        return False


def evaluate_equation(equation_str) -> int | None:
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError('Invalid characters in equation.')

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {'__builtins__': None}, {})
        return result
    except Exception:
        return None


def generate_candidate_expression(num_terms, rng):
    numbers = [rng.randint(100) for _ in range(num_terms)]
    syms = symbols(f'x:{num_terms}')
    expr = syms[0]

    for i in range(1, num_terms):
        op = rng.choice(['+', '-', '*', '/'])
        if op == '+':
            expr = expr + syms[i]
        elif op == '-':
            expr = expr - syms[i]
        elif op == '*':
            expr = expr * syms[i]
        else:  # division
            if numbers[i] != 0:
                current = int(expr.subs({sym: num for sym, num in zip(syms[:i], numbers[:i])}))
                remaining = [n for n in numbers[i:] if n != 0]
                rng.shuffle(remaining)
                found_divisor = False
                for div in remaining:
                    if current % div == 0:
                        numbers[i] = div
                        expr = expr / syms[i]
                        found_divisor = True
                        break
                if not found_divisor:
                    expr = expr - syms[i]
            else:
                expr = expr + syms[i]

    return expr, numbers, syms


def generate_expression(rng, num_terms=4):
    max_attempts = 100
    for attempt in range(max_attempts):
        try:
            expr, numbers, syms = generate_candidate_expression(num_terms, rng)

            # Substitute actual numbers to get target
            subs = {sym: num for sym, num in zip(syms, numbers)}
            target = int(expr.subs(subs))

            # Convert to string expression
            expr_str = str(expr)
            for i, sym in enumerate(syms):
                expr_str = expr_str.replace(str(sym), str(numbers[i]))

            # Ensure target is within bounds
            if target > 0 and target <= 1000:
                return expr_str, numbers, target

        except (ValueError, ZeroDivisionError):
            continue

    raise ValueError(f'Failed to generate valid expression after {max_attempts} attempts')


def score_countdown_action(numbers, target, action_msg, format_reward):
    """Returns (reward, evaluated_answer, is_valid_equation)."""
    is_valid = valid_equation_for_numbers(numbers, action_msg)
    reward = 0.0
    evaluated = None
    if is_valid:
        reward = float(format_reward)
        evaluated = evaluate_equation(extract_xml_answer(action_msg))
        if evaluated is not None and abs(evaluated - target) < 1e-6:
            reward = 1.0
    return reward, evaluated, is_valid


class CountdownEnv(BanditEnv):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**{**config, **kwargs})
        self.tokenizer = tokenizer
        if not self.env_nickname:
            self.env_nickname = f'countdown-{self.num_terms}-{self.seqlen_str()}'

    def reset(self, idx):
        rng = np.random.RandomState(idx)
        _, numbers, target = generate_expression(rng, self.num_terms)
        rng.shuffle(numbers)
        output_tokens = self.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': USER_PROMPT.format(target=target, numbers=numbers),
                },
            ],
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )
        state = CountdownState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            numbers=numbers,
            correct_answer=target,
        )
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
        action_msg = self.tokenizer.decode(action_tokens)
        is_max_tokens = len(action_tokens) >= self.tokens_per_action
        reward, evaluated_answer, is_valid = score_countdown_action(
            state.numbers, state.correct_answer, action_msg, self.format_reward
        )
        if is_max_tokens and is_valid:
            reward -= float(self.max_tokens_penalty)

        render_str = '\n'.join(
            [
                f'{self.tokenizer.decode(state.tokens + action_tokens)}',
                f'Evaluated answer: {evaluated_answer}',
                f'Correct answer: {state.correct_answer}',
                f'Valid equation? {is_valid}',
                f'Reward: {reward:.2f}',
            ]
        )
        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        state = replace(state, tokens=state.tokens + action_tokens, logprobs=new_logprobs, rendered=render_str)
        return (
            state,
            [],
            reward,
            True,
            {
                'valid_equation': is_valid,
                'correct_answer': reward >= 1.0,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
            },
        )
