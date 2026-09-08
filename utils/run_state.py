import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import jax
from jax.experimental.multihost_utils import sync_global_devices

from lmpo.utils.gcs_utils import file_exists, gcs_uri, get_gcs_bucket, list_dir, pkl_load, pkl_save, smart_open


@dataclass
class RunState:
    step: int = 0
    train_iter: int = 0
    total_rollouts: int = 0
    total_rollout_iters: int = 0
    env_task_idx: int = 0
    rng: Any = None
    wandb_run_id: str = ''
    elapsed_seconds: float = 0.0
    performance_meter_state: dict = field(default_factory=dict)
    phase: str = ''
    # Per-iter rollout state (only meaningful when phase == 'sampling').
    num_rollout_iters: int = 0
    env_infos_history: dict = field(default_factory=dict)
    sampling_infos_history: dict = field(default_factory=dict)
    reward_infos_history: dict = field(default_factory=dict)


def checkpoint_dir(wandb_dir, step):
    return os.path.join(wandb_dir, 'checkpoints', f'step_{step:06d}')


def latest_checkpoint_dir(wandb_dir):
    ckpt_root = os.path.join(wandb_dir, 'checkpoints')
    for name in reversed(sorted(d for d in list_dir(ckpt_root) if d.startswith('step_'))):
        d = os.path.join(ckpt_root, name)
        if file_exists(os.path.join(d, 'COMMIT')):
            return d
    return None


def orbax_path(save_dir):
    base = gcs_uri(save_dir) if get_gcs_bucket(save_dir) else save_dir
    return f'{base}/train_state'


def has_orbax_train_state(load_dir):
    base = os.path.join(load_dir, 'train_state')
    return any(file_exists(os.path.join(base, m)) for m in ('manifest.ocdbt', '_CHECKPOINT_METADATA', 'checkpoint'))


def save_checkpoint(save_dir, run_state, train_state, rollouts_buffer):
    """Synchronous save. Writes train_state, run_state, rollouts_buffer, then COMMIT marker."""
    if get_gcs_bucket(save_dir) is None:
        os.makedirs(save_dir, exist_ok=True)

    if train_state is not None:
        import orbax.checkpoint as ocp

        ckptr = ocp.StandardCheckpointer()
        ckptr.save(orbax_path(save_dir), train_state)
        ckptr.wait_until_finished()
        ckptr.close()

    sync_global_devices('ckpt_write')

    if jax.process_index() == 0:
        pkl_save(run_state, os.path.join(save_dir, 'run_state.pkl'))
        pkl_save(dict(rollouts_buffer), os.path.join(save_dir, 'rollouts_buffer.pkl'))
        with smart_open(os.path.join(save_dir, 'COMMIT'), 'wb') as f:
            f.write(b'')


def load_checkpoint(load_dir, train_state_template=None):
    print(f'Loading run_state.pkl from {load_dir}')
    run_state = pkl_load(os.path.join(load_dir, 'run_state.pkl'))

    buf_path = os.path.join(load_dir, 'rollouts_buffer.pkl')
    if file_exists(buf_path):
        print(f'Loading rollouts_buffer.pkl from {load_dir}')
        buf = defaultdict(list, pkl_load(buf_path))
    else:
        buf = defaultdict(list)

    train_state_data = None
    if has_orbax_train_state(load_dir):
        assert train_state_template is not None, 'orbax checkpoint requires train_state_template'
        print(f'Loading orbax train_state from {load_dir}')
        import orbax.checkpoint as ocp

        train_state_data = ocp.StandardCheckpointer().restore(orbax_path(load_dir), target=train_state_template)
    sync_global_devices('train_state_load')

    print(f'Loaded checkpoint: step={run_state.step}, train_iter={run_state.train_iter}')
    return run_state, train_state_data, buf
