"""Small W&B initialization helper with checkpoint-owned resume identity."""

import os
from pathlib import Path
import sys
import threading
import time

import jax
import wandb

from lmpo.utils.configs import to_plain
from lmpo.utils.sharding import broadcast_str_process0

LMPO_PATH = Path(__file__).resolve().parents[1]


def prefix_wandb_name(config):
    name = config.wandb_name or ''
    env_nickname = config.env.env_nickname if config.get('env') else ''
    if name and env_nickname and env_nickname not in name:
        config.wandb_name = f'{env_nickname}-{name}'


def start_debug_log_watchdog(run_dir, max_bytes=1_000_000):
    """Exit if W&B enters an unbounded debug-log retry loop."""
    log_path = os.path.join(run_dir, 'logs', 'debug.log')

    def watch():
        while True:
            try:
                if os.path.getsize(log_path) > max_bytes:
                    sys.stderr.write(f'wandb debug.log exceeded {max_bytes} bytes at {log_path}, exiting.\n')
                    sys.stderr.flush()
                    os._exit(1)
            except OSError:
                pass
            time.sleep(10)

    threading.Thread(target=watch, daemon=True).start()


def init_wandb(config, extra_flags, resume_id=None):
    """Initialize W&B on host 0 and broadcast its directory and run ID."""
    entity = None
    run_id = None
    run_dir = None
    prefix_wandb_name(config)

    if jax.process_index() == 0:
        init_kwargs = {
            'config': {**to_plain(config), **extra_flags},
            'project': config.wandb_project,
            'group': config.wandb_group or None,
            'name': config.wandb_name or None,
            'dir': str(LMPO_PATH),
            'mode': 'online' if config.wandb_online else 'offline',
            'save_code': False,
        }
        if resume_id:
            init_kwargs.update(id=resume_id, resume='allow')

        run = wandb.init(**init_kwargs)
        entity = run.entity or ''
        run_id = run.id
        run_dir = run.dir

        start_debug_log_watchdog(run.dir)

    entity = broadcast_str_process0(entity)
    run_id = broadcast_str_process0(run_id)
    run_dir = broadcast_str_process0(run_dir)

    if config.wandb_online and entity:
        print(f'Run URL: https://wandb.ai/{entity}/{config.wandb_project}/runs/{run_id}')
    print('Files directory:', run_dir)
    return run_dir, run_id
