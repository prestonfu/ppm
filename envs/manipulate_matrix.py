import re
from dataclasses import dataclass, field, replace

from lmpo.envs.base_math import BaseMathEnv, MathWithOracleState, base_math_config


config = {
    **base_math_config,
    'tokens_per_action': 2048,
    'eval_num_tasks': 512,
    'matrix_size': 6,
    'n_ops': 5,
}

SYSTEM_PROMPT = (
    'Provide the final matrix in <answer> </answer> tags, with one row per line and values separated by spaces:\n'
    '<answer>\n'
    '1 2\n'
    '4 5\n'
    '</answer>'
)


def extract_xml_answer(text):
    if '<answer>' not in text or '</answer>' not in text:
        return ''
    return text.split('<answer>')[-1].split('</answer>')[0].strip()


def has_formatting(text):
    return '<answer>' in text and '</answer>' in text


def matrix_to_str(m):
    return '\n'.join(' '.join(str(x) for x in row) for row in m)


def matrix_to_row_labeled_str(m):
    return '\n'.join(f'Row {i + 1}: {" ".join(str(x) for x in row)}' for i, row in enumerate(m))


def normalize_matrix_str(text):
    rows = [' '.join(line.split()) for line in text.strip().splitlines() if line.strip()]
    return '\n'.join(rows)


def _op_description(op):
    t = op['transform']
    descriptions = {
        'hmirror': 'flip horizontally',
        'vmirror': 'flip vertically',
        'dmirror': 'transpose along main diagonal',
        'cmirror': 'transpose along anti-diagonal',
    }
    if t in descriptions:
        return descriptions[t]
    if t == 'rotate':
        return f'rotate {op["degrees"]}°'
    if t == 'map':
        return f'map value {op["from"]} → {op["to"]}'
    if t == 'zero_divisible':
        return f'zero out cells divisible by {op["k"]}'
    if t == 'crop':
        return f'crop rows {op["row_start"]}:{op["row_end"]}, cols {op["col_start"]}:{op["col_end"]}'
    if t == 'remove_every_nth_row':
        return f'remove every {op["n"]}th row'
    if t == 'remove_every_nth_col':
        return f'remove every {op["n"]}th column'
    return t


def _clean_desc(op):
    return op.get('instruction', _op_description(op)).strip().lstrip('- ').rstrip('.')


def generate_rubric(operations, steps, final_matrix):
    items = []
    for i, (op, step) in enumerate(zip(operations, steps)):
        desc = _clean_desc(op)
        step_str = matrix_to_str(step)
        items.append(f'{i + 1}. After step {i + 1} ({desc}), the matrix is:\n{step_str}')
    final_str = matrix_to_str(final_matrix)
    items.append(f'{len(operations) + 1}. The final answer is enclosed in <answer> </answer> tags.')
    items.append(
        f'{len(operations) + 2}. The content inside <answer> </answer> matches the matrix (rows separated by newlines, columns by spaces):\n{final_str}'
    )
    return '\n\n'.join(items), len(items)


def generate_solution(operations, steps, final_matrix):
    lines = ['Let me apply each operation in order.\n']
    for i, (op, step) in enumerate(zip(operations, steps)):
        lines.append(f'Step {i + 1}: {_clean_desc(op)}')
        lines.append(matrix_to_row_labeled_str(step))
        lines.append('')
    lines += ['</think>', '', '<answer>', matrix_to_str(final_matrix), '</answer>']
    return '\n'.join(lines)


@dataclass(frozen=True)
class ManipulateMatrixState(MathWithOracleState):
    problem: str = field(default='', metadata={'gather': True})
    step_strs: tuple = field(default_factory=tuple)


_INSTRUCTION_REWRITES = {
    '- Horizontally mirror the matrix': '- Mirror the matrix over the horizontal axis',
    '- Vertically mirror the matrix': '- Mirror the matrix over the vertical axis',
}


def _rewrite_instruction(instr):
    if instr in _INSTRUCTION_REWRITES:
        return _INSTRUCTION_REWRITES[instr]
    m = re.match(r'(- Rotate the matrix )(\d+)( degrees)', instr)
    if m:
        return f'{m.group(1)}{m.group(2)}{m.group(3)} counterclockwise'
    return instr


def _rewrite_example(row):
    for op in row['metadata'].get('operations', []):
        if 'instruction' in op:
            op['instruction'] = _rewrite_instruction(op['instruction'])
    for old, new in _INSTRUCTION_REWRITES.items():
        row['question'] = row['question'].replace(old, new)
    row['question'] = re.sub(
        r'(- Rotate the matrix )(\d+)( degrees)(?! counterclockwise)',
        r'\1\2\3 counterclockwise',
        row['question'],
    )
    return row


def _apply_steps(ds, matrix, operations):
    from copy import deepcopy

    dispatch = {
        'hmirror': lambda op: ds._hmirror(matrix),
        'vmirror': lambda op: ds._vmirror(matrix),
        'dmirror': lambda op: ds._dmirror(matrix),
        'cmirror': lambda op: ds._cmirror(matrix),
        'rotate': lambda op: ds._rotations[op['degrees']](matrix),
        'map': lambda op: ds._map(matrix, op['from'], op['to']),
        'zero_divisible': lambda op: ds._zero_divisible(matrix, op['k']),
        'crop': lambda op: ds._crop(matrix, op['row_start'], op['row_end'], op['col_start'], op['col_end']),
        'remove_every_nth_row': lambda op: ds._remove_every_nth_row(matrix, op['n']),
        'remove_every_nth_col': lambda op: ds._remove_every_nth_col(matrix, op['n']),
    }
    steps = []
    for op in operations:
        matrix = dispatch[op['transform']](op)
        steps.append(deepcopy(matrix))
    return steps


def _generate_example(matrix_size, n_ops, idx):
    from copy import deepcopy
    from reasoning_gym.algorithmic.manipulate_matrix import ManipulateMatrixDataset, ManipulateMatrixConfig

    ds = ManipulateMatrixDataset(
        ManipulateMatrixConfig(
            seed=idx,
            size=1,
            min_rows=matrix_size,
            max_rows=matrix_size,
            min_cols=matrix_size,
            max_cols=matrix_size,
            min_transforms=n_ops,
            max_transforms=n_ops,
        )
    )
    # reasoning_gym bug: config w_crop/w_zero_divisible are swapped relative to _all_transforms order
    w = ds._weights.copy()
    w[6], w[7] = w[7], w[6]
    # zero out crop after correcting the swap
    for i, t in enumerate(ds._all_transforms):
        if t == 'crop':
            w[i] = 0.0
    ds._weights = w / w.sum()
    row = ds[0]
    row['question'] = row['question'].replace('- Identity transformation, i.e. no change\n', '')
    row['metadata']['steps'] = _apply_steps(ds, deepcopy(row['metadata']['matrix']), row['metadata']['operations'])
    return _rewrite_example(row)


class ManipulateMatrixEnv(BaseMathEnv):
    state_cls = ManipulateMatrixState
    slug = 'manipulate_matrix'

    def __init__(self, tokenizer, matrix_size=6, n_ops=5, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.matrix_size = matrix_size
        self.n_ops = n_ops
        if not self.env_nickname:
            self.env_nickname = f'matrix{matrix_size}x{n_ops}-{self.seqlen_str()}'

    def load_dataset(self):
        self.num_tasks = -1

    def correct_answer(self, idx):
        raise NotImplementedError

    def system_prompt(self):
        return SYSTEM_PROMPT

    def chat_messages(self, question):
        return [{'role': 'system', 'content': self.system_prompt()}, {'role': 'user', 'content': question}]

    def reset(self, idx):
        row = _generate_example(self.matrix_size, self.n_ops, idx)
        question = row['question']
        operations = row['metadata'].get('operations', [])
        steps = row['metadata'].get('steps', [])
        final_matrix = steps[-1] if steps else []

        rubric, rubric_n_items = generate_rubric(operations, steps, final_matrix)
        solution = generate_solution(operations, steps, final_matrix)
        output_tokens = self.tokenizer.apply_chat_template(
            self.chat_messages(question),
            add_generation_prompt=True,
            enable_thinking=bool(int(self.enable_thinking)),
        )

        return ManipulateMatrixState(
            prompt_text=self.tokenizer.decode(output_tokens),
            tokens=output_tokens,
            logprobs=[0] * len(output_tokens),
            correct_answer=matrix_to_str(final_matrix),
            oracle_solution=solution,
            rubric=rubric,
            rubric_n_items=rubric_n_items,
            problem=question,
            step_strs=tuple(matrix_to_row_labeled_str(s) for s in steps),
        ), output_tokens

    def step(self, state, action_tokens, action_logprobs=None):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.eos_token_id)
        action_msg = self.tokenizer.decode(action_tokens)
        is_max_tokens = len(action_tokens) >= self.tokens_per_action

        predicted_str = extract_xml_answer(action_msg)
        reward = 0.0
        correct = False
        if has_formatting(action_msg):
            reward = self.format_reward
            if normalize_matrix_str(predicted_str) == normalize_matrix_str(state.correct_answer):
                reward = 1.0
                correct = True

        gt_step_grades = [int(s in action_msg) for s in state.step_strs]
        rubric_grades = gt_step_grades + [int(has_formatting(action_msg)), int(correct)]
        rubric_score = sum(rubric_grades) / max(len(rubric_grades), 1)

        render_str = '\n'.join(
            [
                self.tokenizer.decode(state.tokens + action_tokens),
                f'Predicted: {predicted_str}',
                f'Expected: {state.correct_answer}',
                f'Correct?: {correct}',
                f'Reward: {reward:.2f}',
            ]
        )
        if action_logprobs is not None:
            action_logprobs = action_logprobs[: len(action_tokens)]
        new_logprobs = state.logprobs + (action_logprobs if action_logprobs is not None else [0] * len(action_tokens))
        state = replace(state, tokens=state.tokens + action_tokens, logprobs=new_logprobs, rendered=render_str)
        n_rubric_items = len(rubric_grades)
        return (
            state,
            [],
            reward,
            True,
            {
                'has_formatting': reward > 0.0,
                'correct_answer': correct,
                'action_length': len(action_tokens),
                'is_max_tokens': is_max_tokens,
                'env_dense_reward': rubric_score,
                **{
                    f'env_dense_reward_part_{i + 1}_of_{n_rubric_items}': rubric_grades[i]
                    for i in range(n_rubric_items)
                },
            },
        )


if __name__ == '__main__':
    import os
    import sys
    from lmpo.models.tokenizer import create_tokenizer

    args = {'idx': '0', 'tokenizer_dir': '/gcs/jaxconverted/Qwen3-0.6B/'}
    for arg in sys.argv[1:]:
        key, value = arg.split('=', 1)
        args[key] = value

    tokenizer = create_tokenizer(os.path.expanduser(args['tokenizer_dir']))
    env = ManipulateMatrixEnv(
        tokenizer=tokenizer,
        matrix_size=config['matrix_size'],
        n_ops=config['n_ops'],
    )

    idx = int(args['idx'])
    state, _ = env.reset(idx)
    row = _generate_example(config['matrix_size'], config['n_ops'], idx=idx)
    solution = generate_solution(
        row['metadata']['operations'],
        row['metadata']['steps'],
        row['metadata']['steps'][-1],
    )

    print('=== Question ===\n', state.problem)
    print('=== Answer ===\n', state.correct_answer)
    print('=== Rubric ===\n', state.rubric, f'\n({state.rubric_n_items} items)')
    print('=== Synthetic Solution ===\n', solution)

    solution_tokens = tokenizer.encode(solution)
    _, _, reward, _, info = env.step(state, solution_tokens)

    print('=== Verification ===')
    reward_keys = sorted(
        [k for k in info if k.startswith('env_dense_reward_part_')],
        key=lambda k: int(k.split('_')[4]),
    )
    for k in reward_keys:
        print(f'  {k}: {"PASS" if info[k] else "FAIL"}')
    n = len(reward_keys)
    total = sum(info[k] for k in reward_keys)
    print(f'  gt_rubric_score: {total / n:.3f} ({total}/{n})')
