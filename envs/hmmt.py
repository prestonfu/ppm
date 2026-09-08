from lmpo.envs.base_math import (
    BaseMathEnv,
    BaseMathState,
    base_math_config,
)


config = {**base_math_config}


class Hmmt2025State(BaseMathState):
    pass


class Hmmt2025Env(BaseMathEnv):
    state_cls = Hmmt2025State
    slug = 'hmmt2025'

    def load_dataset(self) -> None:
        from datasets import load_dataset

        self.ds = load_dataset('MathArena/hmmt_feb_2025', split='train')
        self.num_tasks = len(self.ds)

    def correct_answer(self, idx) -> str:
        return str(self.ds[idx]['answer'])

    def chat_messages(self, idx) -> list[dict]:
        return [
            {'role': 'system', 'content': self.system_prompt()},
            {'role': 'user', 'content': self.ds[idx]['problem']},
        ]
