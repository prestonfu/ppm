import dataclasses
import gc
import math
import sys
from collections import defaultdict

import jax
import numpy as np
from lmpo.core.metrics.common import ChunkedValue
from lmpo.core.metrics.runner import RewardMetrics
from lmpo.core.rollouts import append_rollout_table_rows, collect_rollouts
from lmpo.core.sampling import Sampler
from lmpo.envs.env_creator import create_env
from lmpo.envs.multitask import MultiTaskEnv
from lmpo.models.qwen3 import create_model_from_ckpt, get_tp_spec
from lmpo.models.tokenizer import create_tokenizer
from lmpo.utils.configs import (
    load_config,
    normalize_env_config,
    normalize_reward_config,
)
from lmpo.utils.logging import wandb_table_from_rows
from lmpo.utils.performance_meter import PerformanceMeter
from lmpo.utils.sharding import create_sharding, get_shard_params_fn
from lmpo.utils.statistics import append_pass_at_k, pass_at_k_confidence_intervals
from lmpo.utils.wandb_utils import init_wandb
from tqdm import tqdm

import wandb


class Evaluator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        shard_data_fn,
        params_shard,
        no_shard,
        data_shard,
        get_local_slice,
        inference_batch_per_device,
        tp_size=1,
        reward_meter=None,
        gemini_budget=float('inf'),
        allow_prompt_truncation=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shard_data_fn = shard_data_fn
        self.no_shard = no_shard
        self.data_shard = data_shard
        self.get_local_slice = get_local_slice
        self.inference_batch_per_device = inference_batch_per_device
        self.tp_size = tp_size
        self.reward_meter = reward_meter
        self.gemini_budget = gemini_budget
        self.allow_prompt_truncation = allow_prompt_truncation
        self.shard_params = get_shard_params_fn(params_shard, no_shard)

    def eval(self, params, env, num_epochs, sampler=None, verbose=False):
        sub_envs = env.envs if isinstance(env, MultiTaskEnv) else [env]
        all_metrics = {}
        rows_by_env = {}
        for sub_env in sub_envs:
            metrics, sampler, rows = self.eval_single_env(params, sub_env, num_epochs, sampler, verbose)
            all_metrics.update({f'{sub_env.env_nickname}/{name}': metric for name, metric in metrics.items()})
            rows_by_env[sub_env.env_nickname] = rows
        return all_metrics, sampler, rows_by_env

    def eval_single_env(self, params, env, num_epochs, sampler=None, verbose=False):
        env_num_tasks = env.num_tasks if env.num_tasks != -1 else 512
        total_rollouts = int(num_epochs * env_num_tasks)
        rollout_batch_size = jax.device_count() // self.tp_size * self.inference_batch_per_device
        num_batches = math.ceil(total_rollouts / rollout_batch_size)
        selected_tasks = set(range(min(4, env_num_tasks)))
        obs_length = env.obs_length if int(env.obs_length) != -1 else env.prompt_length

        metrics = defaultdict(list)
        task_idxs = []
        table_rows = []
        env_task_idx = 0
        rng = jax.random.PRNGKey(jax.process_index())
        sampling_params = self.shard_params(params)
        if sampler is None:
            sampler = Sampler(self.model, self.tokenizer, self.data_shard, self.no_shard)

        for batch_idx in tqdm(range(num_batches), desc=f'eval {env.env_nickname}', disable=not verbose):
            rollout, rng, env_task_idx = collect_rollouts(
                params=sampling_params,
                env=env,
                rng=rng,
                sampler=sampler,
                env_task_idx=env_task_idx,
                group_size=1,
                rollout_batch_size=rollout_batch_size,
                prompt_length=env.prompt_length,
                obs_length=obs_length,
                num_generation_tokens=env.tokens_per_action,
                force_answer_at=env.force_answer_at,
                force_end_think_at=env.force_end_think_at,
                allow_prompt_truncation=self.allow_prompt_truncation,
                shard_data=self.shard_data_fn,
                get_local_slice=self.get_local_slice,
                num_tasks=env_num_tasks,
                verbose=verbose and batch_idx == 0,
            )
            keep = min(rollout_batch_size, total_rollouts - len(task_idxs))
            per_turn_returns = rollout.env_returns_by_turn[:keep]
            batch_metrics = {'env_return': env.get_traj_return(per_turn_returns).astype(np.float32)}
            if env.num_turns > 1:
                for turn in range(env.num_turns):
                    batch_metrics[f'turn{turn}/env_reward'] = per_turn_returns[:, turn]
            for name, info in rollout.env_infos.items():
                info = np.asarray(info)[:keep]
                if env.num_turns == 1:
                    batch_metrics[name] = info[:, 0]
                else:
                    for turn in range(env.num_turns):
                        batch_metrics[f'turn{turn}/{name}'] = info[:, turn]

            reward_metrics, reward_extras = {}, {}
            if self.reward_meter:
                if self.reward_meter.has_separate_model:
                    sampling_params = None
                    gc.collect()
                turn_metrics, reward_extras, rng = self.reward_meter.evaluate_rollout(
                    rollout,
                    base_params=params,
                    sampling_params=sampling_params,
                    sampler=sampler,
                    rng=rng,
                    num_rollouts=keep,
                    total_rollout_iters=batch_idx,
                    prompt_length=env.prompt_length,
                    tokens_per_action=env.tokens_per_action,
                    run_all_metrics=True,
                    verbose=verbose and batch_idx == 0,
                )
                for name, metric in turn_metrics.items():
                    value = metric.values if isinstance(metric, ChunkedValue) else metric
                    reduced = np.asarray(value).sum(axis=-1)
                    reward_metrics[name] = reduced.mean(axis=-1)
                    if env.num_turns > 1:
                        for turn in range(env.num_turns):
                            reward_metrics[f'turn{turn}/{name}'] = reduced[:, turn]
                if sampling_params is None:
                    sampling_params = self.shard_params(params)
            batch_metrics.update(reward_metrics)
            for name, metric in batch_metrics.items():
                metrics[name].extend(np.asarray(metric).tolist())
            task_idxs.extend(np.asarray(rollout.env_task_idxs[:keep]).tolist())
            append_rollout_table_rows(table_rows, env, rollout, keep, selected_tasks, batch_metrics, reward_extras)

            if PerformanceMeter.total('gemini_cost/') >= self.gemini_budget:
                print(
                    f'[eval {env.env_nickname}] Gemini budget of ${self.gemini_budget} exceeded. Stopping.', flush=True
                )
                break

        pass_history = defaultdict(list)
        append_pass_at_k(pass_history, metrics['env_return'], task_idxs, 'env_return_')
        if env.num_turns > 1:
            for turn in range(env.num_turns):
                name = f'turn{turn}/env_reward'
                append_pass_at_k(pass_history, metrics[name], task_idxs, f'{name}_')
        for name, values in pass_history.items():
            metrics[name] = [float(np.mean(values))]
        for name, value in pass_at_k_confidence_intervals(pass_history, prefix='').items():
            metrics[name] = [value]
        metrics = {name: np.asarray(metric) for name, metric in metrics.items()}
        if verbose:
            mean_return = float(np.mean(metrics['env_return']))
            print(f'[eval {env.env_nickname}] n={len(task_idxs)} env_return={mean_return:.4f}')
        return metrics, sampler, table_rows


if __name__ == '__main__':
    cfg, extra_flags = load_config(
        {
            'wandb_project': 'lmpo-eval',
            'wandb_name': 'eval',
            'wandb_group': '',
            'wandb_online': 1,
            'model_dir': '/gcs/jaxconverted/Qwen3-1.7B/',
            'debug': 0,
            'num_epochs': 1,
            'allow_prompt_truncation': 1,
            'gemini_budget': 100,
            'diagnostic_every_rollout_iters': 50,
            'save_dir': '',
            'reward': {
                'metrics': {},
                'diagnostic_interval': 50,
                'weights': {},
            },
            'env': {'env_name': 'poem', 'env_nickname': ''},
            'sampling': {
                'inference_batch_per_device': 64,
                'use_flash_attn': 1,
                'use_remat': 0,
                'fsdp': 0,
                'tp_size': 1,
            },
        },
        sys.argv,
    )
    cfg.env = normalize_env_config(cfg.env)
    cfg = normalize_reward_config(cfg)
    if cfg.debug:
        from lmpo.utils.debug import enable_debug

        enable_debug()
        if 'debug' not in cfg.wandb_name:
            cfg.wandb_name += '_debug'

    tokenizer = create_tokenizer(cfg.model_dir)
    env = create_env(cfg.env, tokenizer)
    print(cfg)

    model, params = create_model_from_ckpt(
        cfg.model_dir,
        use_remat=bool(cfg.sampling.use_remat),
        use_flash_attn=bool(cfg.sampling.use_flash_attn),
        mesh=None,
        tp_size=cfg.sampling.tp_size,
    )
    params_shard, no_shard, data_shard, shard_data, get_local_slice = create_sharding(
        params, bool(cfg.sampling.fsdp), cfg.sampling.tp_size, get_tp_spec
    )
    sampling_model = dataclasses.replace(model, mesh=data_shard.mesh)
    reward_meter = None
    if cfg.reward.metrics:
        reward_meter = RewardMetrics(
            cfg, env, tokenizer, sampling_model, params_shard, None, data_shard, no_shard, get_local_slice, shard_data
        )
    print('Reward metrics:', reward_meter)
    init_wandb(cfg, extra_flags)

    evaluator = Evaluator(
        sampling_model,
        tokenizer,
        shard_data_fn=shard_data,
        params_shard=params_shard,
        no_shard=no_shard,
        data_shard=data_shard,
        get_local_slice=get_local_slice,
        inference_batch_per_device=cfg.sampling.inference_batch_per_device,
        tp_size=cfg.sampling.tp_size,
        reward_meter=reward_meter,
        gemini_budget=cfg.gemini_budget,
        allow_prompt_truncation=bool(cfg.allow_prompt_truncation),
    )
    metrics, _, rows_by_env = evaluator.eval(params, env, cfg.num_epochs, verbose=True)

    first_row = next((rows[0] for rows in rows_by_env.values() if rows), None)
    if first_row:
        print(' ======================= Example Rollout ======================= ')
        print(first_row['text'])
        print(' =============================================================== ')
    if jax.process_index() == 0:
        payload = {f'eval/{name}': float(np.mean(metric)) for name, metric in metrics.items()}
        payload.update({f'perf/{name}': value for name, value in PerformanceMeter.get().items()})
        payload.update(
            {
                f'eval_rollouts_table/{nickname}': wandb_table_from_rows(rows)
                for nickname, rows in rows_by_env.items()
                if rows
            }
        )
        wandb.log(payload)
        wandb.finish()
