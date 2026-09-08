from dataclasses import dataclass, replace
import numpy as np
from lmpo.envs.base import BaseEnv, BaseState

config = {
    'env_list': [],  # list of env config dicts, 'weight'
    'env_nickname': '',
    'shuffle_seed': 0,
}


@dataclass
class MultiTaskState:
    env_id: int
    inner_state: BaseState

    def __getattr__(self, name):
        inner_state = self.__dict__.get('inner_state')
        if inner_state is None:
            raise AttributeError(name)
        return getattr(inner_state, name)


class MultiTaskEnv(BaseEnv):
    def __init__(self, envs, env_nickname='', weights=None, shuffle_seed=0):
        super().__init__(
            tokens_per_action=None,
            force_end_think_at=None,
            force_answer_at=None,
        )  # unused
        assert len(envs) > 0, 'must provide at least one env'
        self.envs = envs
        self.n = len(envs)
        self.shuffle_seed = int(shuffle_seed)
        sub_num_turns = {env.num_turns for env in envs}
        assert len(sub_num_turns) == 1, f'sub-envs must share num_turns, got {sub_num_turns}'
        self.num_turns = next(iter(sub_num_turns))
        self.set_env_nickname(env_nickname)
        self.set_mixture(weights)

    def set_env_nickname(self, env_nickname):
        self.env_nicknames = env_nicknames = [env.env_nickname for env in self.envs]
        assert len(set(env_nicknames)) == self.n, f'env_nicknames {env_nicknames} must be unique'
        if not env_nickname:
            env_nickname = '-'.join(env_nicknames)
        self.env_nickname = env_nickname

    def set_mixture(self, weights):
        """Repeat each sub-env's tasks according to ``weights`` and permute."""
        if not weights or any(w is None for w in weights):
            print(f'WARNING: weights is None or contains None, using default weights')
            weights = [1] * self.n
        assert len(weights) == self.n
        self.weights = weights
        env_ids = [np.full(env.num_tasks, env_id) for env_id, env in enumerate(self.envs)]
        env_ids = np.concatenate([np.repeat(x, w) for x, w in zip(env_ids, weights)])
        self.num_tasks = len(env_ids)
        rng = np.random.default_rng(self.shuffle_seed)
        perm = rng.permutation(self.num_tasks)
        self.env_ids_by_task = env_ids[perm].astype(np.int32)
        env_task_idxs = [np.arange(env.num_tasks) for env in self.envs]
        env_task_idxs = np.concatenate([np.repeat(x, w) for x, w in zip(env_task_idxs, weights)])
        self.env_task_idxs_by_task = env_task_idxs[perm].astype(np.int32)

    def get_mixture(self):
        return self.weights.copy()

    def sub_env_ids(self, task_idxs):
        return self.env_ids_by_task[np.asarray(task_idxs, dtype=np.int32)]

    def sub_env_task_idxs(self, task_idxs):
        return self.env_task_idxs_by_task[np.asarray(task_idxs, dtype=np.int32)]

    def state_classes(self):
        return [state_cls for env in self.envs for state_cls in env.state_classes()]

    def __repr__(self):
        return f'MultiTaskEnv(envs={self.envs}, weights={self.weights})'

    def reset(self, idx):
        env_id = int(self.env_ids_by_task[idx])
        env_task_idx = int(self.env_task_idxs_by_task[idx])
        inner_state, output_tokens = self.envs[env_id].reset(env_task_idx)
        multitask_state = MultiTaskState(env_id=env_id, inner_state=inner_state)
        return multitask_state, output_tokens

    def step(self, state: MultiTaskState, action_tokens, **kwargs):
        env_id = state.env_id
        next_inner_state, traj_tokens, reward, done, info = self.envs[env_id].step(
            state.inner_state, action_tokens, **kwargs
        )
        next_state = replace(state, inner_state=next_inner_state)
        return next_state, traj_tokens, reward, done, info

    def get_traj_return(self, per_turn_rewards):
        return self.envs[0].get_traj_return(per_turn_rewards)

    def render(self, state):
        env_id = state.env_id
        return self.envs[env_id].render(state.inner_state)
