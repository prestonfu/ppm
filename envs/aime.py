"""AIME 2024 (HuggingFaceH4) and AIME 2025 (opencompass), both on ``BaseMathEnv``."""

from lmpo.envs.base_math import (
    BaseMathEnv,
    BaseMathState,
    base_math_config,
)

aime2024_config = {**base_math_config}
aime2025_config = {**base_math_config}
# Default for `from lmpo.envs.aime import config` (same as aime2024)
config = aime2024_config


class AimeState(BaseMathState):
    pass


class Aime2024Env(BaseMathEnv):
    state_cls = AimeState
    slug = 'aime2024'

    def load_dataset(self) -> None:
        from datasets import load_dataset

        self.ds = load_dataset('HuggingFaceH4/aime_2024', split='train')
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx) -> str:
        return str(self.ds[idx]['answer'])

    def chat_messages(self, idx) -> list[dict]:
        return [
            {'role': 'system', 'content': self.system_prompt()},
            {'role': 'user', 'content': self.ds[idx]['problem']},
        ]


class Aime2025Env(BaseMathEnv):
    state_cls = AimeState
    slug = 'aime2025'

    def load_dataset(self) -> None:
        from datasets import concatenate_datasets, load_dataset

        ds1 = load_dataset('opencompass/AIME2025', 'AIME2025-I', split='test')
        ds2 = load_dataset('opencompass/AIME2025', 'AIME2025-II', split='test')
        self.ds = concatenate_datasets([ds1, ds2])
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx) -> str:
        return str(self.ds[idx]['answer'])

    def chat_messages(self, idx) -> list[dict]:
        return [
            {'role': 'system', 'content': self.system_prompt()},
            {'role': 'user', 'content': self.ds[idx]['question']},
        ]
