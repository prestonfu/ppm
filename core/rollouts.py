import dataclasses

import jax
import numpy as np
from lmpo.envs.base import (
    gather_env_states,
    gather_step_infos,
    state_extra_fields,
    stack_step_infos,
)
from lmpo.utils.array_utils import pad_and_collate
from lmpo.utils.sharding import host_gather


@dataclasses.dataclass
class RolloutBatch:
    tokens: np.ndarray
    sampler_logprobs: np.ndarray
    is_forced: np.ndarray
    roles: np.ndarray
    action_idx: np.ndarray
    env_states: list
    env_returns_by_turn: np.ndarray
    env_infos: dict
    env_task_idxs: np.ndarray
    prompt_truncated: np.ndarray


def append_rollout_table_rows(rows, env, rollout, keep, selected_tasks, metrics, extras):
    for row_idx, task_idx in enumerate(rollout.env_task_idxs[:keep]):
        if int(task_idx) not in selected_tasks:
            continue
        state = rollout.env_states[row_idx]
        row = {
            'env_nickname': env.env_nickname,
            'env_task_idx': int(task_idx),
            **state_extra_fields(state),
            'text': env.render(state),
        }
        row.update({name: np.asarray(metric)[row_idx] for name, metric in metrics.items()})
        row.update({name: str(np.asarray(extra, dtype=object)[row_idx]) for name, extra in extras.items()})
        rows.append(row)


def collect_rollouts(
    params,
    env,
    rng,
    sampler,
    env_task_idx,
    group_size,
    rollout_batch_size,
    prompt_length,
    obs_length,
    num_generation_tokens,
    force_answer_at,
    force_end_think_at,
    allow_prompt_truncation,
    shard_data,
    get_local_slice,
    num_tasks=None,
    verbose=False,
):
    """Collect one globally materialized batch of grouped, multi-turn rollouts."""
    if rollout_batch_size % group_size:
        raise ValueError(f'rollout_batch_size={rollout_batch_size} must be divisible by group_size={group_size}')

    tokenizer = sampler.tokenizer
    env_num_tasks = num_tasks if num_tasks is not None else env.num_tasks if env.num_tasks != -1 else 1_000_000
    num_prompts = rollout_batch_size // group_size
    env_task_idxs = np.arange(env_task_idx, env_task_idx + num_prompts) % env_num_tasks
    env_task_idxs = np.repeat(env_task_idxs, group_size).astype(np.int32)
    env_task_idx = (env_task_idx + num_prompts) % env_num_tasks

    env_task_idxs_local = np.asarray(get_local_slice(env_task_idxs), dtype=np.int32)
    env_states_local = []
    prompt_lists_local = []
    for task_idx in env_task_idxs_local.tolist():
        state, prompt = env.reset(task_idx)
        env_states_local.append(state)
        prompt_lists_local.append(prompt)

    prompt_tokens_local, prompt_truncated_local = pad_and_collate(
        prompt_lists_local,
        pad_id=tokenizer.pad_token_id,
        force_length=prompt_length,
        description='env turn 0 prompt tokens',
        raise_on_truncation=not allow_prompt_truncation,
        verbose=verbose,
    )
    prompt_truncated_local = prompt_truncated_local.astype(bool)
    prompt_tokens_sharded = shard_data(prompt_tokens_local)

    num_turns = int(env.num_turns)
    max_seq_len = prompt_length + num_generation_tokens + (obs_length + num_generation_tokens) * (num_turns - 1)
    sampling_state = None
    env_returns_by_turn_local = np.zeros(
        (len(env_states_local), num_turns),
        dtype=np.float32,
    )
    step_infos_by_turn = []

    for turn in range(num_turns):
        rng, sample_key = jax.random.split(rng)
        sampling_state = sampler.sample(
            params,
            prompt_tokens_sharded,
            num_generation_tokens,
            sample_key,
            state=sampling_state,
            max_seq_len=max_seq_len,
            force_answer_at=force_answer_at,
            force_end_think_at=force_end_think_at,
            verbose=verbose,
            timer_name='sampling',
        )
        action_idx = int(jax.device_get(sampling_state.next_action_idx)) - 1
        action_tokens_local, action_logprobs_local = sampler.get_local_action_tokens(
            sampling_state,
            action_idx,
            include_logprobs=True,
        )
        (
            env_states_local,
            next_prompts_local,
            rewards_local,
            dones_local,
            step_infos_local,
        ) = env.step_list(
            env_states_local,
            action_tokens_local,
            action_logprobs=action_logprobs_local,
        )
        env_returns_by_turn_local[:, turn] = np.asarray(
            rewards_local,
            dtype=np.float32,
        )
        step_infos_by_turn.append(
            gather_step_infos(
                step_infos_local,
                len(env_states_local),
                shard_data,
            )
        )

        if turn < num_turns - 1:
            prompt_tokens_local, prompt_truncated_next = pad_and_collate(
                next_prompts_local,
                pad_id=tokenizer.pad_token_id,
                force_length=obs_length,
                description=f'env turn {turn + 1} obs tokens',
                raise_on_truncation=not allow_prompt_truncation,
                verbose=verbose and turn == 0,
            )
            prompt_truncated_local |= prompt_truncated_next.astype(bool)
            prompt_tokens_sharded = shard_data(prompt_tokens_local)
        elif not all(dones_local):
            raise ValueError('all environments must be done after the final turn')

    tokens = np.asarray(host_gather(sampling_state.tokens))
    sampler_logprobs = np.asarray(host_gather(sampling_state.logprobs))
    is_forced = np.asarray(host_gather(sampling_state.is_forced))
    roles = np.asarray(host_gather(sampling_state.role))
    action_idx = np.asarray(host_gather(sampling_state.action_idx))
    env_states = gather_env_states(env_states_local, shard_data)
    env_returns_by_turn = np.asarray(host_gather(shard_data(env_returns_by_turn_local)))
    prompt_truncated = np.asarray(host_gather(shard_data(prompt_truncated_local.astype(np.float32)))).astype(bool)
    env_task_idxs = np.asarray(host_gather(shard_data(env_task_idxs_local)))
    env_infos = stack_step_infos(step_infos_by_turn, rollout_batch_size)

    return (
        RolloutBatch(
            tokens=tokens,
            sampler_logprobs=sampler_logprobs,
            is_forced=is_forced,
            roles=roles,
            action_idx=action_idx,
            env_states=env_states,
            env_returns_by_turn=env_returns_by_turn,
            env_infos=env_infos,
            env_task_idxs=env_task_idxs,
            prompt_truncated=prompt_truncated,
        ),
        rng,
        env_task_idx,
    )
