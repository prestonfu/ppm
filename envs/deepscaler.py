"""From https://github.com/agentica-project/rllm and https://github.com/agentica-project/rllm/blob/main/rllm/rewards/math_utils/utils.py"""

from lmpo.envs.base_math import (
    BASE_MATH_SYSTEM_PROMPT,
    BaseMathEnv,
    BaseMathState,
    base_math_config,
)

config = {**base_math_config}


class DeepscalerState(BaseMathState):
    pass


class DeepscalerEnv(BaseMathEnv):
    state_cls = DeepscalerState
    slug = 'deepscaler'

    def load_dataset(self) -> None:
        from datasets import load_dataset

        self.ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')['train']
        if self.shuffle:
            self.ds = self.ds.shuffle(seed=42)
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx) -> str:
        return self.ds[idx]['answer']

    def chat_messages(self, idx) -> list[dict]:
        return [
            {'role': 'system', 'content': BASE_MATH_SYSTEM_PROMPT},
            {'role': 'user', 'content': self.ds[idx]['problem']},
        ]
