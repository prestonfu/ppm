from lmpo.envs.base_math import (
    BASE_MATH_SYSTEM_PROMPT,
    BaseMathEnv,
    BaseMathState,
    base_math_config,
)

config = {**base_math_config}


class DAPOMath17KState(BaseMathState):
    pass


class DAPOMath17KEnv(BaseMathEnv):
    state_cls = DAPOMath17KState
    slug = 'dapo'

    def load_dataset(self):
        from datasets import load_dataset

        self.ds = load_dataset('open-r1/DAPO-Math-17k-Processed')['train']
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx) -> str:
        return self.ds[idx]['solution']

    def chat_messages(self, idx) -> list[dict]:
        row = self.ds[idx]
        return [
            {'role': 'system', 'content': BASE_MATH_SYSTEM_PROMPT},
            {'role': 'user', 'content': row['prompt']},
        ]
