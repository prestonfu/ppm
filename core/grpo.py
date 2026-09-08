import dataclasses
import gc
import sys
import time
from collections import defaultdict
from functools import partial

import jax

jax.distributed.initialize()
print(f'Process {jax.process_index()} of {jax.process_count()}')

import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import NamedSharding, PartitionSpec

from lmpo.core.eval import Evaluator
from lmpo.core.metrics.common import ChunkedValue
from lmpo.core.metrics.runner import RewardMetrics
from lmpo.core.opsd import (
    align_base_teacher_hidden,
    build_opsd_training_batch,
    loss_opsd,
    teacher_topk,
    token_divergence,
)
from lmpo.core.rollouts import collect_rollouts
from lmpo.core.sampling import Sampler, TokenRole
from lmpo.envs.env_creator import create_env
from lmpo.models.qwen3 import create_model_from_ckpt, get_tp_spec
from lmpo.models.tokenizer import create_tokenizer
from lmpo.utils.configs import (
    conf_default_to,
    load_config,
    normalize_env_config,
    normalize_reward_config,
    parse_wallclock_interval,
)
from lmpo.utils.logging import crossed_multiple, prefix_metrics, wandb_table_from_rows
from lmpo.utils.performance_meter import PerformanceMeter, model_flops_per_token
from lmpo.utils.jax_utils import init_jax_compilation_cache
from lmpo.utils.run_state import (
    RunState,
    checkpoint_dir,
    latest_checkpoint_dir,
    load_checkpoint,
    save_checkpoint,
)
from lmpo.utils.sharding import create_sharding, get_shard_params_fn, host_gather
from lmpo.utils.statistics import (
    append_pass_at_k,
    pass_at_k_confidence_intervals,
)
from lmpo.utils.timer import Timer
from lmpo.utils.train_state import TrainState
from lmpo.utils.wandb_utils import init_wandb


def load_grpo_config():
    config, extra_flags = load_config(
        {
            'wandb_project': 'lmpo',
            'wandb_name': 'debug',
            'wandb_group': '',
            'wandb_online': 1,
            'model_dir': '/gcs/jaxconverted/Qwen3-1.7B/',
            'do_save': 0,
            'save_dir': '',
            'save_every_steps': 50,
            'save_every_wallclock': '1:00:00',
            'log_every_steps': 1,
            'print_every_steps': 50,
            'render_every_rollouts': 10000,
            'eval_every_steps': 50,
            'eval_every_wallclock': '6:00:00',
            'diagnostic_every_rollout_iters': 50,
            'max_runtime': '24:00:00',
            'max_steps': 500,
            'max_iters': 10000,
            'gemini_budget': 100,
            'allow_prompt_truncation': 1,
            'env': {'env_name': '', 'env_nickname': ''},
            'test_env': {'env_name': '', 'env_nickname': '', 'num_epochs': 1},
            'reward': {
                'metrics': {},
                'diagnostic_interval': 50,
                'weights': {
                    'env': 1.0,
                    'env_dense_reward': 0.0,
                },
            },
            'sampling': {
                'inference_batch_per_device': 64,
                'use_flash_attn': 1,
                'fsdp': 0,
                'tp_size': 1,
            },
            'test_sampling': {
                'inference_batch_per_device': 64,
                'use_flash_attn': 1,
                'fsdp': 0,
                'tp_size': 1,
            },
            'train': {
                'groups_per_batch': 256,
                'group_size': 8,
                'ppo_minibatch': 64,
                'ppo_microbatch': -1,
                'logprob_minibatch': -1,
                'logit_chunks': 1,
                'do_group_normalization': 1,
                'do_std_normalization': 1,
                'do_global_normalization': 0,
                'do_per_turn_advantage': 0,
                'do_mean_turn_advantage': 0,
                'do_group_filter': 1,
                'do_clip_advantages': 0,
                'do_mask_inference_ratio': 0,
                'do_mask_importance_ratio': 0,
                'do_mask_forced_tokens': 1,
                'do_mask_zero_advantages': 0,
                'do_length_filter': 0,
                'negative_advantage_multiplier': 1.0,
                'lr': 1e-6,
                'clip_low': 0.2,
                'clip_high': 0.2,
                'entropy_coef': 0.001,
                'kl_loss_coef': 0.0,
                'kl_penalty': '',
                'pg_coef': 1.0,
                'opsd_coef': 0.0,
                'opsd': {
                    'alpha': 0.5,
                    'distillation_topk': 100,
                    'distillation_add_tail': 1,
                    'is_clip': 2.0,
                    'prompt_length': -1,
                    'audit_model_name': 'gemini-2.5-flash-lite',
                    'audit_num_workers': 16,
                },
                'weight_decay': 1e-2,
                'train_vocab': 1,
                'fsdp': 1,
                'tp_size': 1,
                'use_remat': 1,
                'use_flash_attn': 1,
            },
        },
        sys.argv,
    )
    config.env = normalize_env_config(config.env)
    if config.test_env.env_name:
        config.test_env = normalize_env_config(config.test_env)
    config.train.opsd.prompt_length = conf_default_to(config.train.opsd.prompt_length, config.env.prompt_length)
    config = normalize_reward_config(config)
    return config, extra_flags


def mean_history_metrics(histories, prefix):
    return {f'{prefix}{name}': float(np.mean(history)) for name, history in histories.items() if history}


def log_rollout_table(
    step,
    env,
    task_idxs,
    states,
    env_returns,
    train_returns,
    reward_renders,
    table_rows,
    max_tasks=2,
):
    if jax.process_index() != 0:
        return

    new_rows = []
    for task_idx in np.unique(task_idxs)[:max_tasks]:
        row_idxs = np.flatnonzero(task_idxs == task_idx)
        group_returns = env_returns[row_idxs]
        for row_idx in row_idxs:
            row = {
                'step': step,
                'env_nickname': env.env_nickname,
                'env_task_idx': int(task_idx),
                'text': env.render(states[row_idx]),
                'env_return': float(env_returns[row_idx]),
                'env_return_mean': float(group_returns.mean()),
                'env_return_std': float(group_returns.std()),
                'env_return_list': group_returns.tolist(),
                'train_return': float(train_returns[row_idx]),
                **{f'reward/{name}': str(values[row_idx]) for name, values in reward_renders.items()},
            }
            new_rows.append(row)
            table_rows.append(row)
    if not new_rows:
        return

    row = new_rows[0]
    print(f'\n================= Rollout task {row["env_task_idx"]} =================\n')
    print(row['text'])
    for name in reward_renders:
        print(f'\n----- Reward metric: {name} -----\n{row[f"reward/{name}"]}')

    wandb.log({'rollouts_table': wandb_table_from_rows(table_rows)})


def build_token_advantages(per_turn_returns, reward_metrics, env_infos, action_idx, env, config):
    num_turns = per_turn_returns.shape[1]
    chunk_counts = {v.n_chunks for v in reward_metrics.values() if isinstance(v, ChunkedValue) and v.n_chunks > 1}
    assert len(chunk_counts) <= 1, f'Conflicting reward chunk counts: {chunk_counts}'
    n_chunks = next(iter(chunk_counts), 1)

    def as_turn_chunks(x):
        arr = x.values if isinstance(x, ChunkedValue) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        return np.broadcast_to(arr, (arr.shape[0], num_turns, n_chunks)).copy()

    def compute_train_returns():
        env_returns = as_turn_chunks(per_turn_returns) * config.reward.weights.env
        metric_returns = np.zeros_like(env_returns)
        for metric_name, weight in config.reward.weights.items():
            if metric_name == 'env' or weight == 0:
                continue
            if metric_name not in reward_metrics and metric_name not in env_infos:
                raise ValueError(f'reward.weights.{metric_name}={weight} but {metric_name!r} was not produced')
            metric_returns += weight * as_turn_chunks(reward_metrics.get(metric_name, env_infos.get(metric_name)))

        return env_returns + metric_returns

    def compute_group_advantages(train_returns):
        batch_size = train_returns.shape[0]
        if config.train.do_per_turn_advantage:
            advantage_source = train_returns
        else:
            trajectory_returns = (
                train_returns.mean(axis=1)
                if getattr(config.train, 'do_mean_turn_advantage', 0)
                else env.get_traj_return(train_returns).astype(np.float32)
            )
            advantage_source = np.broadcast_to(trajectory_returns[:, None, :], (batch_size, num_turns, n_chunks)).copy()

        advantages = advantage_source.reshape(-1, config.train.group_size, num_turns, n_chunks)
        if config.train.do_group_normalization:
            advantages = advantages - advantages.mean(axis=1, keepdims=True)
            if config.train.do_std_normalization:
                advantages = advantages / (advantages.std(axis=1, keepdims=True) + 1e-8)
        if config.train.do_global_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if config.train.do_clip_advantages:
            advantages = np.clip(advantages, 0, None)
        return advantages

    def compute_token_advantages(group_advantages):
        rollout_advantages = group_advantages.reshape(action_idx.shape[0], num_turns, n_chunks)
        is_action = action_idx >= 0
        turn_idx = np.where(is_action, action_idx, 0).astype(np.int32)
        same_turn = (action_idx[:, :, None] == np.arange(num_turns)) & is_action[:, :, None]
        action_lengths = same_turn.sum(axis=1)
        token_positions = np.take_along_axis(np.cumsum(same_turn, axis=1), turn_idx[:, :, None], axis=2).squeeze(-1) - 1
        token_action_lengths = np.take_along_axis(action_lengths, turn_idx, axis=1).clip(min=1)
        token_chunks = np.clip(token_positions * n_chunks // token_action_lengths, 0, n_chunks - 1)
        batch_idx = np.arange(action_idx.shape[0])[:, None]
        return np.where(is_action, rollout_advantages[batch_idx, turn_idx, token_chunks], 0.0).astype(np.float32)

    train_returns = compute_train_returns()
    group_advantages = compute_group_advantages(train_returns)
    token_advantages = compute_token_advantages(group_advantages)
    return n_chunks, train_returns, group_advantages, token_advantages


def get_kl(policy_logprobs, reference_logprobs, estimator):
    log_ratio = policy_logprobs - reference_logprobs
    k1 = log_ratio
    k2 = 0.5 * jnp.square(log_ratio)
    k3 = jnp.exp(-log_ratio) - 1.0 + log_ratio

    if estimator == 'k1':
        return k1
    if estimator == 'k2':
        return k2
    if estimator == 'k3':
        return k3
    if estimator == 'k1+':
        return k2 + jax.lax.stop_gradient(k1 - k2)
    if estimator == 'k3+':
        return k2 + jax.lax.stop_gradient(k3 - k2)
    raise ValueError(f'Unknown KL estimator: {estimator}')


def create_grpo_ops(
    config,
    train_data_sharding,
    train_state_sharding,
    pad_id,
    opsd_enabled,
):
    """Build the compiled numerical operations used by the training loop."""
    logits_sharding = (
        NamedSharding(
            train_data_sharding.mesh,
            PartitionSpec('data', None, 'model'),
        )
        if config.train.tp_size > 1
        else None
    )

    def target_tokens(tokens):
        return jnp.concatenate(
            [
                tokens[:, 1:],
                jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype),
            ],
            axis=-1,
        )

    def constrain_logits(logits):
        logits = logits.astype(jnp.float32)
        if logits_sharding is not None:
            logits = jax.lax.with_sharding_constraint(logits, logits_sharding)
        return logits

    def chunk_slices(sequence_length):
        n_chunks = int(config.train.logit_chunks)
        if sequence_length % n_chunks:
            raise ValueError(f'sequence length {sequence_length} is not divisible by logit_chunks={n_chunks}')
        chunk_size = sequence_length // n_chunks
        return [slice(index * chunk_size, (index + 1) * chunk_size) for index in range(n_chunks)]

    @jax.jit
    def get_logprobs(train_state, params, token_ids):
        token_mask = (token_ids != pad_id).astype(jnp.int32)
        hidden, _ = train_state.call_model(
            token_ids,
            token_mask,
            cache=None,
            params=params,
            return_hidden=True,
        )
        targets = target_tokens(token_ids)
        lm_kernel = params['Dense_0']['kernel']

        parts = []
        for token_slice in chunk_slices(hidden.shape[1]):
            logits = constrain_logits(hidden[:, token_slice] @ lm_kernel)
            log_norm = jax.nn.logsumexp(logits, axis=-1)
            selected = jnp.take_along_axis(
                logits,
                targets[:, token_slice, None],
                axis=-1,
            )[..., 0]
            parts.append(selected - log_norm)
        return jnp.concatenate(parts, axis=1)[:, :-1]

    def get_trainer_logprobs(train_state, token_ids):
        return get_logprobs(train_state, train_state.params, token_ids)

    def loss_fn(
        grad_params,
        train_state,
        base_params,
        token_ids,
        action_loss_mask,
        token_advantages,
        trainer_logprobs,
        sampler_logprobs,
        is_max_tokens,
        opsd_tokens=None,
        opsd_teacher_pos=None,
        opsd_mask=None,
    ):
        targets = target_tokens(token_ids)
        token_mask = (token_ids != pad_id).astype(jnp.int32)
        action_loss_mask = action_loss_mask[:, 1:]
        sampler_logprobs = sampler_logprobs[:, 1:]

        if not config.train.train_vocab:
            grad_params['Dense_0']['kernel'] = jax.lax.stop_gradient(grad_params['Dense_0']['kernel'])
            grad_params['Embed_0']['embedding'] = jax.lax.stop_gradient(grad_params['Embed_0']['embedding'])

        hidden, _ = train_state.call_model(
            token_ids,
            token_mask,
            cache=None,
            params=grad_params,
            return_hidden=True,
        )

        teacher_hidden = teacher_kernel = None
        if opsd_enabled:
            teacher_hidden, teacher_kernel = align_base_teacher_hidden(
                train_state,
                base_params,
                opsd_tokens,
                opsd_teacher_pos,
                pad_id,
            )

        @jax.remat
        def project_chunk(
            hidden_chunk,
            lm_kernel,
            target_chunk,
            teacher_topk_logprobs,
            teacher_topk_indices,
        ):
            logits = constrain_logits(hidden_chunk @ lm_kernel)
            log_norm = jax.nn.logsumexp(logits, axis=-1)
            selected_logits = jnp.take_along_axis(
                logits,
                target_chunk[..., None],
                axis=-1,
            )[..., 0]
            logprobs = selected_logits - log_norm
            entropy = log_norm - jnp.sum(
                jax.nn.softmax(logits) * logits,
                axis=-1,
            )
            divergence = None
            if opsd_enabled:
                divergence = token_divergence(
                    logits,
                    log_norm,
                    teacher_topk_logprobs,
                    teacher_topk_indices,
                    alpha=config.train.opsd.alpha,
                    add_tail=bool(config.train.opsd.distillation_add_tail),
                )
            return logprobs, entropy, divergence

        logprob_parts = []
        entropy_parts = []
        opsd_parts = []
        for token_slice in chunk_slices(hidden.shape[1]):
            teacher_logprobs = teacher_indices = None
            if opsd_enabled:
                teacher_logprobs, teacher_indices = teacher_topk(
                    teacher_hidden[:, token_slice],
                    teacher_kernel,
                    topk=int(config.train.opsd.distillation_topk),
                    logits_sharding=logits_sharding,
                )
            logprobs, entropy, opsd_divergence = project_chunk(
                hidden[:, token_slice],
                grad_params['Dense_0']['kernel'],
                targets[:, token_slice],
                teacher_logprobs,
                teacher_indices,
            )
            logprob_parts.append(logprobs)
            entropy_parts.append(entropy)
            if opsd_enabled:
                opsd_parts.append(opsd_divergence)

        token_logprobs = jnp.concatenate(logprob_parts, axis=1)[:, :-1]
        entropy = jnp.concatenate(entropy_parts, axis=1)[:, :-1]
        opsd_divergence = jnp.concatenate(opsd_parts, axis=1)[:, :-1] if opsd_enabled else None

        old_logprobs = trainer_logprobs
        sampler_to_trainer_ratio = jnp.exp(sampler_logprobs - trainer_logprobs)
        advantages = token_advantages[:, 1:]
        if config.train.negative_advantage_multiplier != 1.0:
            advantages = jnp.where(
                advantages < 0,
                advantages * config.train.negative_advantage_multiplier,
                advantages,
            )

        log_ratio = token_logprobs - old_logprobs
        ratio = jnp.exp(log_ratio)
        unclipped_pg = -advantages * ratio
        clipped_pg = -advantages * jnp.clip(
            ratio,
            1 - config.train.clip_low,
            1 + config.train.clip_high,
        )
        pg_loss_per_token = jnp.maximum(unclipped_pg, clipped_pg)

        mask = action_loss_mask
        if config.train.do_mask_inference_ratio:
            mask *= jnp.abs(sampler_logprobs - trainer_logprobs) < jnp.log(2.0)
        sampler_ratio_filtered_mask = mask
        if config.train.do_mask_importance_ratio:
            mask *= jnp.abs(ratio - 1) < 1.0
        importance_filtered_mask = mask
        if config.train.do_length_filter:
            mask *= (~is_max_tokens.astype(jnp.bool_))[:, None]
        if config.train.do_mask_zero_advantages:
            mask *= advantages != 0

        def masked_mean(value, value_mask=mask):
            return jnp.sum(value * value_mask) / (jnp.sum(value_mask) + 1e-8)

        if config.train.do_mask_zero_advantages:
            grpo_pg = masked_mean(pg_loss_per_token)
        else:
            grpo_pg = jnp.mean(pg_loss_per_token * mask)
        loss_pg = config.train.pg_coef * grpo_pg
        loss_entropy = -config.train.entropy_coef * masked_mean(entropy)
        loss = loss_pg + loss_entropy

        extra_losses = {}
        if opsd_enabled:
            opsd_loss = loss_opsd(
                opsd_divergence,
                token_logprobs,
                sampler_logprobs,
                mask,
                opsd_mask,
                config.train.opsd.is_clip,
            )
            loss += config.train.opsd_coef * opsd_loss
            extra_losses['loss_opsd'] = opsd_loss

        if config.train.kl_loss_coef != 0.0:
            base_hidden, _ = train_state.call_model(
                token_ids,
                token_mask,
                cache=None,
                params=base_params,
                return_hidden=True,
            )
            base_logits = constrain_logits(
                jax.lax.stop_gradient(base_hidden[:, :-1]) @ base_params['Dense_0']['kernel']
            )
            base_log_norm = jax.nn.logsumexp(base_logits, axis=-1)
            base_token_logprobs = (
                jnp.take_along_axis(
                    base_logits,
                    targets[:, :-1, None],
                    axis=-1,
                )[..., 0]
                - base_log_norm
            )
            kl_to_base = masked_mean(
                get_kl(
                    token_logprobs,
                    base_token_logprobs,
                    config.train.kl_penalty,
                )
            )
            loss_kl = config.train.kl_loss_coef * kl_to_base
            loss += loss_kl
            extra_losses.update(loss_kl=loss_kl, kl_base=kl_to_base)

        sampler_trainer_log_ratio = sampler_logprobs - trainer_logprobs
        trainer_to_sampler_ratio = jnp.exp(-sampler_trainer_log_ratio)
        sampler_trainer_probability_difference = jnp.abs(jnp.exp(sampler_logprobs) - jnp.exp(trainer_logprobs)) * mask
        clip_low = ratio - 1 < -config.train.clip_low
        clip_high = ratio - 1 > config.train.clip_high

        return loss, {
            'loss': loss,
            'loss_pg': loss_pg,
            'loss_ent': loss_entropy,
            'grpo_pg': grpo_pg,
            'advantages': masked_mean(advantages),
            'advantages_magnitude': masked_mean(jnp.abs(advantages)),
            'nonzero_advantages': masked_mean(advantages != 0),
            'entropy_per_token': masked_mean(entropy),
            'approx_kl': masked_mean((ratio - 1) - log_ratio),
            'clip_fraction': masked_mean(clip_low | clip_high),
            'clip_fraction_low': masked_mean(clip_low),
            'clip_fraction_high': masked_mean(clip_high),
            'importance_ratio_mean': masked_mean(ratio),
            'importance_ratio_magnitude': masked_mean(jnp.abs(1 - ratio)),
            'inference_recompute/kl': masked_mean((sampler_to_trainer_ratio - 1) - sampler_trainer_log_ratio),
            'inference_recompute/kl_reverse': masked_mean((trainer_to_sampler_ratio - 1) + sampler_trainer_log_ratio),
            'inference_recompute/prob_diff_mean': jnp.mean(sampler_trainer_probability_difference),
            'action_tokens_per_seq': jnp.mean(jnp.sum(action_loss_mask, axis=-1)),
            'ratio_filtered_tokens_per_seq': jnp.mean(jnp.sum(sampler_ratio_filtered_mask, axis=-1)),
            'importance_filtered_tokens_per_seq': jnp.mean(jnp.sum(importance_filtered_mask, axis=-1)),
            'trained_tokens_per_seq': jnp.mean(jnp.sum(mask, axis=-1)),
            'trained_token_fraction': (jnp.sum(mask) / (jnp.sum(action_loss_mask) + 1e-8)),
            'is_max_tokens': jnp.mean(is_max_tokens),
            **extra_losses,
        }

    @partial(
        jax.jit,
        out_shardings=(train_state_sharding.params, None),
        donate_argnums=(0,),
    )
    def accumulate_gradients(
        gradient_sum,
        train_state,
        base_params,
        token_ids,
        action_loss_mask,
        token_advantages,
        trainer_logprobs,
        sampler_logprobs,
        is_max_tokens,
        opsd_tokens=None,
        opsd_teacher_pos=None,
        opsd_mask=None,
    ):
        gradients, info = jax.grad(loss_fn, has_aux=True)(
            train_state.params,
            train_state,
            base_params,
            token_ids,
            action_loss_mask,
            token_advantages,
            trainer_logprobs,
            sampler_logprobs,
            is_max_tokens,
            opsd_tokens,
            opsd_teacher_pos,
            opsd_mask,
        )
        gradient_sum = jax.tree.map(jnp.add, gradient_sum, gradients)
        return gradient_sum, info

    @partial(
        jax.jit,
        out_shardings=(train_state_sharding, None),
        donate_argnums=(0, 1),
        static_argnums=(2,),
    )
    def apply_gradients(train_state, gradients, n_microbatches):
        if n_microbatches != 1:
            scale = jnp.float32(1 / n_microbatches)
            gradients = jax.tree.map(lambda value: value * scale, gradients)
        updates, opt_state = train_state.tx.update(gradients, train_state.opt_state, train_state.params)
        params = optax.apply_updates(train_state.params, updates)
        train_state = train_state.replace(params=params, opt_state=opt_state, step=train_state.step + 1)
        return train_state, {
            'grad_norm': optax.global_norm(gradients),
            'update_norm': optax.global_norm(updates),
            'param_norm': optax.global_norm(params),
        }

    def update(
        train_state,
        base_params,
        token_ids,
        action_loss_mask,
        token_advantages,
        trainer_logprobs,
        sampler_logprobs,
        is_max_tokens,
        opsd_inputs,
    ):
        n_microbatches = token_ids.shape[1]
        gradients = jax.tree.map(jnp.zeros_like, train_state.params)
        microbatch_infos = []

        for microbatch_idx in range(n_microbatches):
            opsd_microbatch = (
                {
                    'opsd_tokens': opsd_inputs['tokens'][:, microbatch_idx],
                    'opsd_teacher_pos': opsd_inputs['teacher_pos'][:, microbatch_idx],
                    'opsd_mask': opsd_inputs['mask'][:, microbatch_idx],
                }
                if opsd_inputs
                else {}
            )
            gradients, info = accumulate_gradients(
                gradients,
                train_state,
                base_params,
                token_ids[:, microbatch_idx],
                action_loss_mask[:, microbatch_idx],
                token_advantages[:, microbatch_idx],
                trainer_logprobs[:, microbatch_idx],
                sampler_logprobs[:, microbatch_idx],
                is_max_tokens[:, microbatch_idx],
                **opsd_microbatch,
            )
            microbatch_infos.append(info)

        train_state, optimizer_info = apply_gradients(train_state, gradients, n_microbatches)
        info = jax.tree.map(lambda *values: sum(values) / len(values), *microbatch_infos)
        info.update(optimizer_info)
        return train_state, info

    return get_trainer_logprobs, update


init_jax_compilation_cache()
config, extra_flags = load_grpo_config()

tokenizer = create_tokenizer(config.model_dir)
pad_id = tokenizer.pad_token_id
env = create_env(config.env, tokenizer)
test_env = create_env(config.test_env, tokenizer) if config.test_env.env_name else None
if test_env is not None:
    test_envs = test_env.envs if hasattr(test_env, 'envs') else [test_env]
    for sub_env in test_envs:
        if sub_env.num_tasks > 512:
            raise ValueError(
                f'test environment {sub_env.env_nickname!r} has {sub_env.num_tasks} tasks; the limit is 512'
            )
print(config)

prompt_length = int(config.env.prompt_length)
obs_length = int(config.env.obs_length) if int(config.env.obs_length) != -1 else prompt_length

flops_per_token, flops_prefill = model_flops_per_token(config.model_dir)
model, params = create_model_from_ckpt(
    config.model_dir,
    use_remat=bool(config.train.use_remat),
    use_flash_attn=bool(config.train.use_flash_attn),
    mesh=None,
    tp_size=config.train.tp_size,
)

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        config.train.lr,
        b1=0.9,
        b2=0.95,
        weight_decay=config.train.weight_decay,
    ),
)
rng = jax.random.PRNGKey(jax.process_index())


def init_shape_fn(rng, params):
    return TrainState.create_with_params(
        rng=rng,
        params=params,
        model_def=model,
        tx=tx,
        use_ema=False,
    )


train_state_shape = jax.eval_shape(init_shape_fn, rng, params)

params_sharding, sample_no_shard, sample_data_sharding, sample_shard, sample_local_slice = create_sharding(
    params,
    fsdp=bool(config.sampling.fsdp),
    tp_size=config.sampling.tp_size,
    get_tp_spec=get_tp_spec,
)
train_state_sharding, _train_no_shard, train_data_sharding, train_shard, train_local_slice = create_sharding(
    train_state_shape,
    fsdp=bool(config.train.fsdp),
    tp_size=config.train.tp_size,
    get_tp_spec=get_tp_spec,
)
test_params_sharding, test_no_shard, test_data_sharding, test_shard, test_local_slice = create_sharding(
    params,
    fsdp=bool(config.test_sampling.fsdp),
    tp_size=config.test_sampling.tp_size,
    get_tp_spec=get_tp_spec,
)

train_model = dataclasses.replace(model, mesh=train_data_sharding.mesh)
sampling_model = dataclasses.replace(
    model,
    mesh=sample_data_sharding.mesh,
    tp_size=config.sampling.tp_size,
    use_flash_attn=bool(config.sampling.use_flash_attn),
)
test_sampling_model = dataclasses.replace(
    model,
    mesh=test_data_sharding.mesh,
    tp_size=config.test_sampling.tp_size,
    use_flash_attn=bool(config.test_sampling.use_flash_attn),
)
test_evaluator = None
if test_env is not None:
    test_evaluator = Evaluator(
        test_sampling_model,
        tokenizer,
        shard_data_fn=test_shard,
        params_shard=test_params_sharding,
        no_shard=test_no_shard,
        data_shard=test_data_sharding,
        get_local_slice=test_local_slice,
        inference_batch_per_device=config.test_sampling.inference_batch_per_device,
        tp_size=config.test_sampling.tp_size,
        allow_prompt_truncation=bool(config.allow_prompt_truncation),
    )
train_state_sharding = train_state_sharding.replace(
    model_def=train_model,
    apply_fn=train_model.apply,
)
train_state = jax.jit(
    lambda rng, params: TrainState.create_with_params(
        rng=rng,
        params=params,
        model_def=train_model,
        tx=tx,
        use_ema=False,
    ),
    out_shardings=train_state_sharding,
)(rng, params)

opsd_enabled = config.train.opsd_coef > 0
if opsd_enabled and int(env.num_turns) != 1:
    raise ValueError(f'OPSD currently supports bandit environments only, got num_turns={env.num_turns}')
base_params = (
    jax.jit(lambda value: value, out_shardings=train_state_sharding.params)(params)
    if config.train.kl_loss_coef != 0.0 or opsd_enabled
    else None
)
shard_sampling_params = get_shard_params_fn(
    params_sharding,
    sample_no_shard,
)
sampler = Sampler(sampling_model, tokenizer, sample_data_sharding, sample_no_shard)
del params

get_trainer_logprobs, update = create_grpo_ops(
    config=config,
    train_data_sharding=train_data_sharding,
    train_state_sharding=train_state_sharding,
    pad_id=pad_id,
    opsd_enabled=opsd_enabled,
)
reward_metrics = None
if config.reward.metrics:
    reward_metrics = RewardMetrics(
        config=config,
        env=env,
        tokenizer=tokenizer,
        model=sampling_model,
        params_shard=params_sharding,
        sampler=sampler,
        data_shard=sample_data_sharding,
        no_shard=sample_no_shard,
        get_local_slice=sample_local_slice,
        shard_data_fn=sample_shard,
    )

rollout_batch_size = jax.device_count() // config.sampling.tp_size * config.sampling.inference_batch_per_device
global_batch_size = config.train.groups_per_batch * config.train.group_size
train_microbatch_size = config.train.ppo_microbatch if config.train.ppo_microbatch != -1 else config.train.ppo_minibatch
logprob_minibatch_size = (
    config.train.logprob_minibatch if config.train.logprob_minibatch != -1 else train_microbatch_size
)
n_minibatches = global_batch_size // config.train.ppo_minibatch
if global_batch_size % config.train.ppo_minibatch:
    raise ValueError(
        f'global_batch_size={global_batch_size} must be divisible by ppo_minibatch={config.train.ppo_minibatch}'
    )
if config.train.ppo_minibatch % train_microbatch_size:
    raise ValueError(
        f'ppo_minibatch={config.train.ppo_minibatch} must be divisible by ppo_microbatch={train_microbatch_size}'
    )
if global_batch_size % logprob_minibatch_size:
    raise ValueError(
        f'global_batch_size={global_batch_size} must be divisible by logprob_minibatch={logprob_minibatch_size}'
    )
if rollout_batch_size % config.train.group_size:
    raise ValueError(
        f'rollout_batch_size={rollout_batch_size} must be divisible by group_size={config.train.group_size}'
    )

step = 0
train_iter = 0
total_rollouts = 0
total_rollout_iters = 0
env_task_idx = 0
test_sampler = None
rollout_table_rows = []
test_rollout_rows_by_env = defaultdict(list)
num_test_evals = 0
resumed_run_state = None
resumed_buffer = None
resumed_elapsed_seconds = 0.0

if config.save_dir:
    resume_dir = latest_checkpoint_dir(config.save_dir)
    if resume_dir:
        resumed_run_state, restored_train_state, resumed_buffer = load_checkpoint(resume_dir, train_state)
        if restored_train_state is not None:
            train_state = restored_train_state
        step = resumed_run_state.step
        train_iter = resumed_run_state.train_iter
        total_rollouts = resumed_run_state.total_rollouts
        total_rollout_iters = getattr(resumed_run_state, 'total_rollout_iters', total_rollouts // rollout_batch_size)
        env_task_idx = resumed_run_state.env_task_idx
        resumed_elapsed_seconds = resumed_run_state.elapsed_seconds
        PerformanceMeter.import_state(resumed_run_state.performance_meter_state)
        if resumed_run_state.rng is not None:
            rng = resumed_run_state.rng

resume_id = resumed_run_state.wandb_run_id if resumed_run_state is not None else None
wandb_dir, wandb_run_id = init_wandb(config, extra_flags, resume_id=resume_id)
save_root = config.save_dir or wandb_dir
start_time = time.time()
last_eval_time = start_time
last_save_time = start_time
eval_time_interval = parse_wallclock_interval(config.eval_every_wallclock)
save_time_interval = parse_wallclock_interval(config.save_every_wallclock)
max_runtime = parse_wallclock_interval(config.max_runtime) if config.max_runtime else 0


def save(phase, buffer, rollout_iter, histories, force=False):
    if not config.do_save or not force:
        return False
    run_state = RunState(
        step=step,
        train_iter=train_iter,
        total_rollouts=total_rollouts,
        total_rollout_iters=total_rollout_iters,
        env_task_idx=env_task_idx,
        rng=np.asarray(rng),
        wandb_run_id=wandb_run_id,
        elapsed_seconds=(resumed_elapsed_seconds + time.time() - start_time),
        performance_meter_state=PerformanceMeter.export_state(),
        phase=phase,
        num_rollout_iters=rollout_iter if phase == 'sampling' else 0,
        env_infos_history=dict(histories['env']) if phase == 'sampling' else {},
        sampling_infos_history=dict(histories['sampling']) if phase == 'sampling' else {},
        reward_infos_history=dict(histories['reward']) if phase == 'sampling' else {},
    )
    new_dir = checkpoint_dir(save_root, step)
    old_dir = latest_checkpoint_dir(save_root)
    save_checkpoint(
        new_dir,
        run_state,
        train_state if step > 0 else None,
        buffer if phase == 'sampling' else {},
    )
    if jax.process_index() == 0 and old_dir and old_dir != new_dir:
        from lmpo.utils.gcs_utils import rm_dir

        rm_dir(old_dir)
    return True


sync_global_devices('after_resume')
runtime_exceeded = False
while step < config.max_steps and train_iter < config.max_iters and not runtime_exceeded:
    buffer = resumed_buffer or defaultdict(list)
    rs = resumed_run_state
    histories = {
        'env': defaultdict(list, rs.env_infos_history if rs else {}),
        'sampling': defaultdict(list, rs.sampling_infos_history if rs else {}),
        'reward': defaultdict(list, rs.reward_infos_history if rs else {}),
    }
    rollout_iter = rs.num_rollout_iters if rs else 0
    resumed_run_state = resumed_buffer = None

    Timer('sampling').start()
    sampling_params = shard_sampling_params(train_state.params)

    while len(buffer['tokens']) < config.train.groups_per_batch:
        rollout, rng, env_task_idx = collect_rollouts(
            params=sampling_params,
            env=env,
            rng=rng,
            sampler=sampler,
            env_task_idx=env_task_idx,
            group_size=config.train.group_size,
            rollout_batch_size=rollout_batch_size,
            prompt_length=prompt_length,
            obs_length=obs_length,
            num_generation_tokens=config.env.tokens_per_action,
            force_answer_at=config.env.force_answer_at,
            force_end_think_at=config.env.force_end_think_at,
            allow_prompt_truncation=bool(config.allow_prompt_truncation),
            shard_data=sample_shard,
            get_local_slice=sample_local_slice,
            verbose=step < 10 and total_rollouts < 10000,
        )

        turns = env.num_turns
        action_tokens = config.env.tokens_per_action
        sample_in = rollout_batch_size * prompt_length * turns
        sample_out = rollout_batch_size * action_tokens * turns
        sampling_flops = flops_prefill(rollout_batch_size, prompt_length) * turns + flops_per_token * sample_out
        PerformanceMeter.add('flops_sampling', sampling_flops)
        PerformanceMeter.add('tokens_in/sampling', sample_in)
        PerformanceMeter.add('tokens_out/sampling', sample_out)

        reward_metrics_by_turn = {}
        reward_renders = {}
        if reward_metrics:
            Timer('reward_metrics').start()
            if reward_metrics.has_separate_model:
                sampling_params = None
                gc.collect()
            try:
                reward_metrics_by_turn, reward_renders, rng = reward_metrics.evaluate_rollout(
                    rollout,
                    base_params=train_state.params,
                    sampling_params=sampling_params,
                    sampler=sampler,
                    rng=rng,
                    num_rollouts=rollout_batch_size,
                    total_rollout_iters=total_rollout_iters,
                    prompt_length=prompt_length,
                    tokens_per_action=config.env.tokens_per_action,
                    audit=total_rollouts == 0,
                    verbose=step < 10 and total_rollouts < 10000,
                )
            finally:
                Timer('reward_metrics').end()

        _, train_returns_by_turn, advantages_grouped, token_advantages = build_token_advantages(
            rollout.env_returns_by_turn,
            reward_metrics_by_turn,
            rollout.env_infos,
            rollout.action_idx,
            env,
            config,
        )
        env_returns = env.get_traj_return(rollout.env_returns_by_turn).astype(np.float32)
        train_returns = (
            train_returns_by_turn.mean(axis=(-1, -2))
            if config.train.do_per_turn_advantage or config.train.do_mean_turn_advantage
            else env.get_traj_return(train_returns_by_turn).mean(axis=-1)
        )

        histories['env']['env_return'].extend(env_returns.tolist())
        histories['env']['train_return'].extend(train_returns.tolist())
        for name, info in rollout.env_infos.items():
            histories['env'][name].extend(np.asarray(info).reshape(-1).tolist())
        for name, metric in reward_metrics_by_turn.items():
            if '/' in name:
                continue
            array = metric.values if isinstance(metric, ChunkedValue) else metric
            histories['reward'][name].extend(array.sum(axis=-1).mean(axis=-1).tolist())
        for name in config.reward.weights:
            if name in rollout.env_infos:
                histories['reward'][name].extend(np.asarray(rollout.env_infos[name]).mean(axis=1).tolist())

        group_ids = np.arange(len(env_returns)) // config.train.group_size
        task_ids = None
        if hasattr(env, 'sub_env_ids'):
            task_ids = env.sub_env_ids(rollout.env_task_idxs)
        append_pass_at_k(histories['env'], env_returns, group_ids, 'env_return_', labels=task_ids)
        if task_ids is not None:
            task_ids = np.asarray(task_ids).astype(np.int32)
            task_names = [sub_env.env_nickname or f'env{index}' for index, sub_env in enumerate(env.envs)]
            for task_id, name in enumerate(task_names):
                mask = task_ids == task_id
                sub_env_returns = env_returns[mask]
                histories['env'][f'{name}/return'].extend(sub_env_returns.tolist())
                append_pass_at_k(histories['env'], sub_env_returns, group_ids[mask], f'{name}/')
        if rollout.env_returns_by_turn.shape[1] > 1:
            for turn in range(rollout.env_returns_by_turn.shape[1]):
                append_pass_at_k(
                    histories['env'], rollout.env_returns_by_turn[:, turn], group_ids, f'turn{turn}/env_reward_'
                )

        n_groups = advantages_grouped.shape[0]

        def group(value):
            return value.reshape(
                n_groups,
                config.train.group_size,
                *value.shape[1:],
            )

        prompt_truncated_groups = group(rollout.prompt_truncated).any(axis=-1)
        zero_advantage_groups = np.all(
            advantages_grouped == 0,
            axis=tuple(range(1, advantages_grouped.ndim)),
        )
        accepted_groups = ~prompt_truncated_groups
        if config.train.do_group_filter:
            accepted_groups &= ~zero_advantage_groups

        if 'is_max_tokens' not in rollout.env_infos:
            raise ValueError('environment must report is_max_tokens in step infos')
        is_max_tokens = rollout.env_infos['is_max_tokens'].any(axis=-1).astype(np.int32)
        histories['sampling']['use_rollouts_rate'].append(float(accepted_groups.mean()))
        histories['env']['prompt_truncated'].append(float(prompt_truncated_groups.mean()))
        histories['sampling']['max_token_rollout_rate'].append(float(is_max_tokens.mean()))
        histories['sampling']['has_forced_tokens'].append(float(np.any(rollout.is_forced, axis=-1).mean()))

        grouped = {
            'tokens': group(rollout.tokens),
            'advantages': group(token_advantages),
            'sampler_logprobs': group(rollout.sampler_logprobs),
            'is_forced': group(rollout.is_forced),
            'roles': group(rollout.roles),
            'is_max_tokens': group(is_max_tokens),
        }

        if opsd_enabled:
            opsd_tokens, opsd_teacher_pos, opsd_mask = build_opsd_training_batch(
                tokenizer,
                env,
                rollout.env_states,
                rollout.tokens,
                rollout.roles,
                rollout.action_idx,
                int(config.train.opsd.prompt_length),
                config.env.tokens_per_action,
                pad_id,
                bool(config.allow_prompt_truncation),
                verbose=step < 10 and total_rollouts < 10000,
                audit=total_rollouts == 0,
                audit_model_name=config.train.opsd.audit_model_name,
                audit_num_workers=config.train.opsd.audit_num_workers,
            )
            grouped['opsd_tokens'] = group(opsd_tokens.astype(np.int32))
            grouped['opsd_teacher_pos'] = group(opsd_teacher_pos.astype(np.int32))
            grouped['opsd_mask'] = group(opsd_mask.astype(np.float32))

        buffer_keys = ['tokens', 'advantages', 'sampler_logprobs', 'is_forced', 'roles', 'is_max_tokens']
        if opsd_enabled:
            buffer_keys += ['opsd_tokens', 'opsd_teacher_pos', 'opsd_mask']
        for group_idx in np.flatnonzero(accepted_groups):
            for key in buffer_keys:
                buffer[key].append(grouped[key][group_idx])

        last_rollout = total_rollouts + rollout_batch_size - 1
        do_render = crossed_multiple(total_rollouts - 1, last_rollout, config.render_every_rollouts)
        if do_render:
            log_rollout_table(
                step=step,
                env=env,
                task_idxs=rollout.env_task_idxs,
                states=rollout.env_states,
                env_returns=env_returns,
                train_returns=train_returns,
                reward_renders=reward_renders,
                table_rows=rollout_table_rows,
            )

        total_rollouts += rollout_batch_size
        rollout_iter += 1
        total_rollout_iters += 1
        do_save = time.time() - last_save_time >= save_time_interval
        if save('sampling', buffer, rollout_iter, histories, force=do_save):
            last_save_time = time.time()
        if PerformanceMeter.total('gemini_cost/') >= config.gemini_budget:
            save('sampling', buffer, rollout_iter, histories, force=True)
            sys.exit(0)
        runtime_exceeded = max_runtime > 0 and time.time() - start_time >= max_runtime
        if runtime_exceeded:
            save('sampling', buffer, rollout_iter, histories, force=True)
            print(f'Max runtime of {config.max_runtime} reached. Stopping.')
            sys.exit(0)

        if sampling_params is None:
            sampling_params = shard_sampling_params(train_state.params)

    Timer('sampling').end()
    sampling_params = None
    gc.collect()

    def clip_buffer(name, dtype=None):
        value = np.concatenate(buffer[name], axis=0)[:global_batch_size]
        return value.astype(dtype) if dtype is not None else value

    token_ids = clip_buffer('tokens')
    token_advantages = clip_buffer('advantages')
    sampler_logprobs = clip_buffer('sampler_logprobs')
    is_forced = clip_buffer('is_forced')
    roles = clip_buffer('roles')
    is_max_tokens = clip_buffer('is_max_tokens')

    action_loss_mask = roles == int(TokenRole.ACTION)
    if config.train.do_mask_forced_tokens:
        action_loss_mask &= ~is_forced

    def ppo_shard(value, batch_size=train_microbatch_size):
        value = jnp.reshape(value, (batch_size, -1, *value.shape[1:]))
        return train_shard(train_local_slice(value))

    ppo_inputs = {
        'token_ids': ppo_shard(token_ids),
        'token_advantages': ppo_shard(token_advantages),
        'sampler_logprobs': ppo_shard(sampler_logprobs),
        'action_loss_mask': ppo_shard(action_loss_mask),
        'is_max_tokens': ppo_shard(is_max_tokens),
    }
    if opsd_enabled:
        ppo_inputs.update(
            {
                'opsd_tokens': ppo_shard(clip_buffer('opsd_tokens', np.int32)),
                'opsd_teacher_pos': ppo_shard(clip_buffer('opsd_teacher_pos', np.int32)),
                'opsd_mask': ppo_shard(clip_buffer('opsd_mask', np.float32)),
            }
        )

    trainer_logprobs = []
    logprob_batches = ppo_shard(token_ids, logprob_minibatch_size)
    Timer('logprobs').start()
    for batch_idx in range(global_batch_size // logprob_minibatch_size):
        batch_tokens = logprob_batches[:, batch_idx]
        batch_logprobs = get_trainer_logprobs(train_state, batch_tokens)
        trainer_logprobs.append(np.asarray(host_gather(batch_logprobs)))
    trainer_logprobs = ppo_shard(np.stack(trainer_logprobs, axis=1).reshape(global_batch_size, -1))
    Timer('logprobs').end()
    PerformanceMeter.add('flops_logprobs', flops_per_token * global_batch_size * token_ids.shape[1])

    for minibatch_idx in range(n_minibatches):
        minibatch = slice(
            minibatch_idx * (config.train.ppo_minibatch // train_microbatch_size),
            (minibatch_idx + 1) * (config.train.ppo_minibatch // train_microbatch_size),
        )
        opsd_minibatch = (
            {
                'tokens': ppo_inputs['opsd_tokens'][:, minibatch],
                'teacher_pos': ppo_inputs['opsd_teacher_pos'][:, minibatch],
                'mask': ppo_inputs['opsd_mask'][:, minibatch],
            }
            if opsd_enabled
            else None
        )
        Timer('update').start()
        train_state, update_info = update(
            train_state=train_state,
            base_params=base_params,
            token_ids=ppo_inputs['token_ids'][:, minibatch],
            action_loss_mask=ppo_inputs['action_loss_mask'][:, minibatch],
            token_advantages=ppo_inputs['token_advantages'][:, minibatch],
            trainer_logprobs=trainer_logprobs[:, minibatch],
            sampler_logprobs=ppo_inputs['sampler_logprobs'][:, minibatch],
            is_max_tokens=ppo_inputs['is_max_tokens'][:, minibatch],
            opsd_inputs=opsd_minibatch,
        )
        Timer('update').end()
        PerformanceMeter.add('flops_train', 3 * flops_per_token * config.train.ppo_minibatch * token_ids.shape[1])

        info = jax.tree.map(lambda value: float(np.asarray(value).mean()), jax.device_get(update_info))
        info = prefix_metrics(info, 'train/')
        info.update(mean_history_metrics(histories['env'], 'env/'))
        info.update(mean_history_metrics(histories['reward'], 'reward/'))
        info.update(mean_history_metrics(histories['sampling'], 'sampling/'))
        info.update(pass_at_k_confidence_intervals(histories['env']))
        info.update(prefix_metrics(Timer.times(), 'time/'))
        info['env_epochs'] = total_rollouts / env.num_tasks if env.num_tasks != -1 else 0
        info['rollout_iters_per_update'] = rollout_iter
        info['total_rollout_iters'] = total_rollout_iters

        runtime_exceeded = max_runtime > 0 and time.time() - start_time >= max_runtime
        is_last_step = (
            runtime_exceeded
            or step >= config.max_steps - 1
            or (train_iter + 1 >= config.max_iters and minibatch_idx == n_minibatches - 1)
        )
        do_print = is_last_step or (config.print_every_steps > 0 and step % config.print_every_steps == 0)
        do_log = is_last_step or (config.log_every_steps > 0 and step % config.log_every_steps == 0)
        if do_print:
            print(f'[{step=}, {total_rollouts=}] Training (iter={train_iter})')
            for name, value in sorted(info.items()):
                print(f'[{step=}, {total_rollouts=}] {name}: {value}')
        log_data = {**info, **PerformanceMeter.get_log_metrics()} if do_log else None
        if log_data is not None and jax.process_index() == 0:
            log_data.update(global_step=step, train_iter=train_iter, total_rollouts=total_rollouts)
            log_data['total_rollout_iters'] = total_rollout_iters
            wandb.log(log_data)
        Timer.reset()
        step += 1
        if runtime_exceeded:
            print(f'Max runtime of {config.max_runtime} reached. Stopping.')
            break

    eval_by_step = crossed_multiple(step - n_minibatches, step, config.eval_every_steps)
    eval_by_time = time.time() - last_eval_time >= eval_time_interval
    do_eval = test_env is not None and (runtime_exceeded or step >= config.max_steps or eval_by_step or eval_by_time)
    if do_eval:
        test_metrics, test_sampler, test_rows = test_evaluator.eval(
            train_state.params,
            test_env,
            config.test_env.num_epochs,
            sampler=test_sampler,
            verbose=num_test_evals == 0,
        )
        num_test_evals += 1
        for nickname, rows in test_rows.items():
            test_rollout_rows_by_env[nickname].extend({'step': step, **row} for row in rows)
        if jax.process_index() == 0:
            payload = {
                'global_step': step,
                **{f'test_env/{name}': float(np.mean(metric)) for name, metric in test_metrics.items()},
            }
            for nickname, rows in test_rollout_rows_by_env.items():
                if rows:
                    payload[f'test_rollouts_table/{nickname}'] = wandb_table_from_rows(rows)
            wandb.log(payload)
        last_eval_time = time.time()

    train_iter += 1
    save_by_step = crossed_multiple(step - n_minibatches, step, config.save_every_steps)
    force_save = runtime_exceeded or step >= config.max_steps or train_iter >= config.max_iters or save_by_step
    saved = save('training', {}, 0, histories, force=force_save)
    if saved:
        last_save_time = time.time()
