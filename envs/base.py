import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass

import numpy as np
from lmpo.utils.sharding import (
    host_gather,
    host_gather_bytes,
    host_gather_strings_by_process,
)


def _type_name(value_type):
    return f'{value_type.__module__}.{value_type.__qualname__}'


def _validate_process_metadata(metadata, context, error=None):
    """Raise the same validation error on every process before a collective."""
    report = json.dumps({'error': error, 'metadata': metadata}, sort_keys=True)
    reports = [json.loads(value) for value in host_gather_strings_by_process([report])]

    errors = [f'process {process}: {value["error"]}' for process, value in enumerate(reports) if value['error']]
    if errors:
        raise ValueError(f'{context} validation failed: {"; ".join(errors)}')

    values = [value['metadata'] for value in reports]
    if any(value != values[0] for value in values[1:]):
        details = '; '.join(f'process {process}: {value}' for process, value in enumerate(values))
        raise ValueError(f'{context} differs across processes: {details}')


def gather_env_states(states_local, shard_data_fn):
    """Gather trusted in-job environment states in mesh data-axis row order."""
    payloads_local = []
    outer_type = None
    local_error = None
    try:
        if not states_local:
            raise ValueError('cannot gather an empty state batch')
        outer_type = type(states_local[0])
        if any(type(state) is not outer_type for state in states_local):
            raise TypeError('local states must have one outer type')
        payloads_local = [pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL) for state in states_local]
    except Exception as exc:
        local_error = f'{type(exc).__name__}: {exc}'

    state_metadata = json.dumps(
        {
            'local_batch_size': len(states_local),
            'outer_type': _type_name(outer_type) if outer_type is not None else '',
        },
        sort_keys=True,
    )
    _validate_process_metadata(
        state_metadata,
        'environment state type',
        local_error,
    )
    payloads_global = host_gather_bytes(payloads_local, shard_data_fn)

    states_global = []
    decode_error = None
    try:
        states_global = [pickle.loads(payload) for payload in payloads_global]
        if any(type(state) is not outer_type for state in states_global):
            raise TypeError('gathered states changed outer type')
    except Exception as exc:
        decode_error = f'{type(exc).__name__}: {exc}'
    _validate_process_metadata(
        state_metadata,
        'environment state decode',
        decode_error,
    )
    return states_global


def _step_info_schema(infos_local, local_batch_size):
    if not isinstance(infos_local, Mapping):
        raise TypeError(f'step infos must be a mapping, got {type(infos_local).__name__}')

    schema = []
    for name in sorted(infos_local):
        values = np.asarray(infos_local[name])
        if values.shape != (local_batch_size,):
            raise ValueError(f'step info {name!r} must have shape ({local_batch_size},), got {values.shape}')
        if not (np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)):
            raise TypeError(f'step info {name!r} must be numeric or boolean, got dtype {values.dtype}')
        schema.append((name, values.dtype.str, values.shape[1:]))
    return json.dumps(
        {'fields': schema, 'local_batch_size': local_batch_size},
        separators=(',', ':'),
        sort_keys=True,
    )


def gather_step_infos(infos_local, local_batch_size, shard_data_fn):
    """Validate and gather scalar step infos in mesh data-axis row order."""
    schema = ''
    local_error = None
    try:
        schema = _step_info_schema(infos_local, local_batch_size)
    except Exception as exc:
        local_error = f'{type(exc).__name__}: {exc}'
    _validate_process_metadata(schema, 'step info schema', local_error)

    infos_global = {}
    for name in sorted(infos_local):
        values_local = np.asarray(infos_local[name])
        infos_global[name] = np.asarray(host_gather(shard_data_fn(values_local)))
    return infos_global


def stack_step_infos(step_infos_by_turn, batch_size):
    """Stack gathered scalar infos into arrays shaped (batch, num_turns)."""
    num_turns = len(step_infos_by_turn)
    keys = sorted({key for turn_infos in step_infos_by_turn for key in turn_infos})
    stacked = {}
    for key in keys:
        values = [np.asarray(turn_infos[key]) for turn_infos in step_infos_by_turn if key in turn_infos]
        dtype = np.result_type(*[value.dtype for value in values])
        result = np.zeros((batch_size, num_turns), dtype=dtype)
        for turn, turn_infos in enumerate(step_infos_by_turn):
            if key not in turn_infos:
                continue
            value = np.asarray(turn_infos[key])
            if value.shape != (batch_size,):
                raise ValueError(f'gathered step info {key!r} must have shape ({batch_size},), got {value.shape}')
            result[:, turn] = value
        stacked[key] = result
    return stacked


def state_extra_fields(state) -> dict:
    exclude = {'tokens', 'logprobs', 'rendered', 'prompt_text'}
    out = {}
    for f in fields(state):
        if f.name in exclude:
            continue
        value = getattr(state, f.name)
        if f.name == 'inner_state' and is_dataclass(value):
            for inner_f in fields(value):
                if inner_f.name not in exclude:
                    out[inner_f.name] = getattr(value, inner_f.name)
            continue
        out[f.name] = value
    return out


def clean_action(action_tokens, end_token, **infos):
    try:
        index = action_tokens.index(end_token)
        return action_tokens[: index + 1], {k: v[: index + 1] for k, v in infos.items()}
    except ValueError:
        return action_tokens, infos


@dataclass(frozen=True)
class BaseState:
    prompt_text: str = ''
    tokens: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    turn: int = 0

    def render(self) -> str:
        raise NotImplementedError


class BaseEnv:
    """Basic class for an LLM RL environment.

    Supports multi-turn rollouts. ``num_turns`` is the number of action
    turns per rollout; ``BanditEnv`` enforces ``num_turns=1``.
    """

    tokens_per_action = 32
    force_answer_at = -1
    num_tasks = -1
    num_turns = 1
    env_nickname = ''
    obs_length = -1  # max length per-turn observation (turn>=1). -1 = fall back to prompt_length.

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def state_classes(self):
        state_cls = getattr(self, 'state_cls', None)
        return [state_cls] if state_cls else []

    def seqlen_str(self):
        s = self.tokens_per_action
        return f'{s // 1024}k' if s >= 1024 else str(s)

    def reset(self, idx):  # return a state, and an initial output_tokens.
        raise NotImplementedError

    def step(self, state, action_tokens, action_logprobs=None, **kwargs):
        # Returns: (state, output_tokens, reward, is_done, infos)
        raise NotImplementedError

    def get_sft_data(self, idx):
        # Returns (tokens, target_mask): prompt + target tokens (no padding), and a
        # 0/1 mask over those tokens marking which positions to train on.
        raise NotImplementedError

    def get_traj_return(self, per_turn_rewards):
        # per_turn_rewards: (B, num_turns) or (B, num_turns, K) -> (B,) or (B, K).
        # Aggregates along axis=1 (the turn axis); any trailing dim is preserved.
        raise NotImplementedError

    def clean_action(self, action_tokens, end_token):
        try:
            index = action_tokens.index(end_token)
            return action_tokens[: index + 1]
        except ValueError:
            return action_tokens

    def step_list(self, states, action_tokens, action_logprobs=None):
        if len(states) != len(action_tokens):
            raise ValueError(
                f'states and action_tokens must have the same length, got {len(states)} and {len(action_tokens)}'
            )
        if action_logprobs is not None and len(action_logprobs) != len(states):
            raise ValueError(f'action_logprobs must have length {len(states)}, got {len(action_logprobs)}')

        new_states = []
        new_output_tokens = []
        new_rewards = []
        new_is_dones = []
        new_infos = {}
        for i, (state, ac) in enumerate(zip(states, action_tokens)):
            lp = action_logprobs[i] if action_logprobs is not None else None
            new_state, output_tokens, reward, is_done, infos = self.step(state, ac, action_logprobs=lp)
            new_states.append(new_state)
            new_output_tokens.append(output_tokens)
            new_rewards.append(reward)
            new_is_dones.append(is_done)
            for k, v in infos.items():
                new_infos.setdefault(k, []).append(v)
        return new_states, new_output_tokens, new_rewards, new_is_dones, new_infos


class BanditEnv(BaseEnv):
    """Single-turn env. Most envs (math, gsm8k, poem, ...) should inherit from this."""

    num_turns = 1

    def __init__(self, **kwargs):
        kwargs.setdefault('num_turns', 1)
        if kwargs['num_turns'] != 1:
            raise ValueError(f'BanditEnv must have num_turns=1, got {kwargs["num_turns"]}.')
        super().__init__(**kwargs)

    def get_traj_return(self, per_turn_rewards):
        assert per_turn_rewards.shape[1] == 1, (
            f'BanditEnv should have per_turn_rewards with shape (B, 1, ...), got {per_turn_rewards.shape}.'
        )
        return per_turn_rewards[:, 0]
