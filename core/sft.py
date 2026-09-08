import dataclasses
import time
import jax


import sys
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from collections import defaultdict
import wandb

from jax.sharding import NamedSharding, PartitionSpec

from lmpo.utils.jax_utils import init_jax_compilation_cache


from lmpo.models.qwen3 import create_model_from_ckpt, get_tp_spec
from lmpo.utils.wandb_utils import init_wandb
from lmpo.envs.env_creator import create_env
from lmpo.utils.sharding import (
    create_sharding,
    host_gather,
    get_memory_usage,
)
from lmpo.utils.train_state import TrainState
from lmpo.models.tokenizer import create_tokenizer
from lmpo.utils.array_utils import pad_and_collate
from lmpo.core.eval import Evaluator
from lmpo.utils.logging import crossed_multiple, prefix_metrics, wandb_table_from_rows
from lmpo.utils.configs import (
    load_config,
    normalize_env_config,
    parse_wallclock_interval,
)
from lmpo.utils.timer import Timer
from lmpo.utils.performance_meter import PerformanceMeter, model_flops_per_token


def get_config():
    cfg, extra_flags = load_config(
        {
            'wandb_project': 'lmpo',
            'wandb_name': 'sft_debug',
            'wandb_group': '',
            'wandb_run_id': '',
            'wandb_online': 1,
            'model_dir': '/gcs/jaxconverted/Qwen3-1.7B/',
            'log_every_steps': 50,
            'print_every_steps': 500,
            'eval_every_steps': 50,
            'eval_every_wallclock': '6:00:00',
            'max_runtime': '24:00:00',
            'max_steps': 500,
            'max_iters': 10000,
            'debug': 0,
            'allow_prompt_truncation': 1,
            'env': {'env_name': 'manipulate_matrix', 'env_nickname': ''},
            'test_env': {'env_name': '', 'env_nickname': '', 'num_epochs': 1},
            'test_sampling': {
                'inference_batch_per_device': 64,
                'use_flash_attn': 1,
                'fsdp': 0,
                'tp_size': 1,
            },
            'train': {
                'global_batch': 32,
                'microbatch': 8,
                'logit_chunks': 1,
                'lr': 2e-5,
                'weight_decay': 1e-2,
                'train_vocab': 1,
                'fsdp': 1,
                'tp_size': 1,
                'use_remat': 1,
                'use_flash_attn': 1,
                'val_fraction': 0.0,
                'val_batches': 4,
                'val_interval': 50,
                'val_seed': 42,
            },
        },
        sys.argv,
    )
    cfg.env = normalize_env_config(cfg.env)
    if cfg.test_env.env_name:
        cfg.test_env = normalize_env_config(cfg.test_env)
    if cfg.debug:
        from lmpo.utils.debug import enable_debug

        enable_debug()
        if 'debug' not in cfg.wandb_name:
            cfg.wandb_name += '_debug'
    return cfg, extra_flags


print(f'Process {jax.process_index()} of {jax.process_count()}')
init_jax_compilation_cache()
config, extra_flags = get_config()

ckpt_dir = config.model_dir
tokenizer = create_tokenizer(ckpt_dir)
pad_id = tokenizer.pad_token_id

env = create_env(config.env, tokenizer)
test_env = create_env(config.test_env, tokenizer) if config.test_env.env_name else None
print(config)

flops_per_token, flops_prefill = model_flops_per_token(ckpt_dir)
model, params = create_model_from_ckpt(
    ckpt_dir,
    use_remat=bool(config.train.use_remat),
    use_flash_attn=bool(config.train.use_flash_attn),
    mesh=None,
    tp_size=config.train.tp_size,
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(config.train.lr, b1=0.9, b2=0.95, weight_decay=config.train.weight_decay),
)
rng = jax.random.PRNGKey(jax.process_index())
print('Memory usage pre-init:', get_memory_usage(), 'GB')
init_shape_fn = partial(TrainState.create_with_params, model_def=model, tx=tx, use_ema=False)
train_state_shape = jax.eval_shape(init_shape_fn, rng=rng, params=params)
train_state_shard, no_shard, train_data_shard, train_shard_data_fn, train_get_local_slice = create_sharding(
    train_state_shape,
    fsdp=bool(config.train.fsdp),
    tp_size=config.train.tp_size,
    get_tp_spec=get_tp_spec,
)
test_params_shard, test_no_shard, test_data_shard, test_shard_data_fn, test_get_local_slice = create_sharding(
    params,
    fsdp=bool(config.test_sampling.fsdp),
    tp_size=config.test_sampling.tp_size,
    get_tp_spec=get_tp_spec,
)
train_model = dataclasses.replace(model, mesh=train_data_shard.mesh)
test_sampling_model = dataclasses.replace(
    model,
    mesh=test_data_shard.mesh,
    tp_size=config.test_sampling.tp_size,
    use_flash_attn=bool(config.test_sampling.use_flash_attn),
)
test_evaluator = None
if test_env is not None:
    test_evaluator = Evaluator(
        test_sampling_model,
        tokenizer,
        shard_data_fn=test_shard_data_fn,
        params_shard=test_params_shard,
        no_shard=test_no_shard,
        data_shard=test_data_shard,
        get_local_slice=test_get_local_slice,
        inference_batch_per_device=config.test_sampling.inference_batch_per_device,
        tp_size=config.test_sampling.tp_size,
    )
init_fn = partial(TrainState.create_with_params, model_def=train_model, tx=tx, use_ema=False)
train_state_shard = train_state_shard.replace(model_def=train_model, apply_fn=train_model.apply)
train_state = jax.jit(lambda r, p: init_fn(rng=r, params=p), out_shardings=train_state_shard)(rng, params)

del params
print('Memory usage train_state:', get_memory_usage(), 'GB')

init_wandb(config, extra_flags, resume_id=config.wandb_run_id or None)

seq_len = config.env.prompt_length + config.env.tokens_per_action


def loss_fn(grad_params, train_state, token_batch, target_mask):
    text_target = jnp.concat(
        [
            token_batch[:, 1:],
            jnp.zeros((token_batch.shape[0], 1), dtype=token_batch.dtype),
        ],
        axis=-1,
    )
    target_mask = target_mask[:, 1:]
    token_mask = jnp.where(token_batch != pad_id, 1, 0).astype(jnp.int32)

    if not config.train.train_vocab:
        grad_params['Dense_0']['kernel'] = jax.lax.stop_gradient(grad_params['Dense_0']['kernel'])
        grad_params['Embed_0']['embedding'] = jax.lax.stop_gradient(grad_params['Embed_0']['embedding'])

    hidden, _ = train_state.call_model(token_batch, token_mask, cache=None, params=grad_params, return_hidden=True)

    logits_sharding = None
    if config.train.tp_size > 1:
        logits_sharding = NamedSharding(train_data_shard.mesh, PartitionSpec('data', None, 'model'))

    @jax.remat
    def head_reduce(hidden_chunk, lm_kernel, target_chunk):
        logits = (hidden_chunk @ lm_kernel).astype(jnp.float32)  # [B, chunk, V]
        if logits_sharding is not None:
            logits = jax.lax.with_sharding_constraint(logits, logits_sharding)
        log_norm = jax.nn.logsumexp(logits, axis=-1)
        token_logits_chunk = jnp.take_along_axis(logits, target_chunk[..., None], axis=-1)[..., 0]
        token_logprobs_chunk = token_logits_chunk - log_norm
        entropy_chunk = log_norm - jnp.sum(jax.nn.softmax(logits) * logits, axis=-1)
        return token_logprobs_chunk, entropy_chunk

    n_logit_chunks = int(config.train.logit_chunks)
    T = hidden.shape[1]
    assert T % n_logit_chunks == 0, f'seqlen {T} not divisible by logit_chunks={n_logit_chunks}'
    chunk_size = T // n_logit_chunks
    logprobs_parts, entropy_parts = [], []
    for i in range(n_logit_chunks):
        sl = slice(i * chunk_size, (i + 1) * chunk_size)
        logprobs_i, ent_i = head_reduce(
            hidden[:, sl],
            grad_params['Dense_0']['kernel'],
            text_target[:, sl],
        )
        logprobs_parts.append(logprobs_i)
        entropy_parts.append(ent_i)
    token_logprobs = jnp.concatenate(logprobs_parts, axis=1)[:, :-1]
    entropy = jnp.concatenate(entropy_parts, axis=1)[:, :-1]

    nll = -token_logprobs
    avg_over_mask = lambda x: jnp.sum(x * target_mask) / (jnp.sum(target_mask) + 1e-8)
    loss = avg_over_mask(nll)

    return loss, {
        'loss': loss,
        'logprob_of_token': loss,
        'entropy_per_token': avg_over_mask(entropy),
        'tokens_deterministic_99': avg_over_mask(jnp.exp(token_logprobs) > 0.99),
        'tokens_deterministic_95': avg_over_mask(jnp.exp(token_logprobs) > 0.95),
        'trained_tokens_per_seq': jnp.mean(jnp.sum(target_mask, axis=-1)),
    }


@partial(jax.jit, out_shardings=(train_state_shard.params, None), donate_argnums=(0,))
def accumulate_grads(grad_accumulator, train_state, token_batch, target_mask):
    print('JIT compiling accumulate_grads for token_batch of shape', token_batch.shape)
    micro_grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params, train_state, token_batch, target_mask)
    new_accumulator = jax.tree.map(jnp.add, grad_accumulator, micro_grads)
    return new_accumulator, info


@partial(jax.jit, out_shardings=(train_state_shard, None), donate_argnums=(0, 1), static_argnums=(2,))
def update_train_step(train_state: TrainState, grads, n_microbatches: int):
    if n_microbatches != 1:
        inv = jnp.float32(1.0 / n_microbatches)
        grads = jax.tree.map(lambda g: g * inv, grads)
    updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    train_state = train_state.replace(params=new_params, opt_state=opt_state, step=train_state.step + 1)
    info = {
        'grad_norm': optax.global_norm(grads),
        'update_norm': optax.global_norm(updates),
        'param_norm': optax.global_norm(new_params),
    }
    return train_state, info


def update(train_state: TrainState, token_batch, target_mask):
    n_microbatches = token_batch.shape[1]
    infos = []
    grads = jax.tree.map(jnp.zeros_like, train_state.params)
    for j in range(n_microbatches):
        grads, micro_info = accumulate_grads(grads, train_state, token_batch[:, j], target_mask[:, j])
        infos.append(micro_info)
    train_state, train_step_info = update_train_step(train_state, grads, n_microbatches)
    info = jax.tree.map(lambda *xs: sum(xs) / len(xs), *infos)
    info.update(train_step_info)
    return train_state, info


@jax.jit
def eval_sft_batch(train_state: TrainState, token_batch, target_mask):
    _, info = loss_fn(train_state.params, train_state, token_batch, target_mask)
    return info


def eval_sft_microbatches(train_state: TrainState, token_batch, target_mask):
    n_microbatches_ = token_batch.shape[1]
    infos = []
    for j in range(n_microbatches_):
        infos.append(eval_sft_batch(train_state, token_batch[:, j], target_mask[:, j]))
    return jax.tree.map(lambda *xs: sum(xs) / len(xs), *infos)


global_batch_size = config.train.global_batch
microbatch_size = config.train.microbatch
assert global_batch_size % microbatch_size == 0, (
    f'global_batch={global_batch_size} not divisible by microbatch={microbatch_size}'
)
n_microbatches = global_batch_size // microbatch_size
data_size = train_data_shard.mesh.shape['data']
assert microbatch_size % data_size == 0, (
    f'microbatch={microbatch_size} must be divisible by data_size={data_size} '
    f'(each FSDP shard needs at least one row per microbatch step)'
)

env_num_tasks = env.num_tasks if env.num_tasks != -1 else 1000000
if env.num_tasks != -1 and float(config.train.val_fraction) > 0:
    all_task_ids = np.arange(env_num_tasks)
    split_rng = np.random.default_rng(int(config.train.val_seed))
    split_rng.shuffle(all_task_ids)
    val_size = max(1, int(env_num_tasks * float(config.train.val_fraction)))
    val_size = min(val_size, env_num_tasks - 1)
    val_task_ids = np.sort(all_task_ids[:val_size])
    train_task_ids = np.sort(all_task_ids[val_size:])
else:
    val_task_ids = np.array([], dtype=np.int64)
    train_task_ids = np.arange(env_num_tasks)
num_train_tasks = len(train_task_ids)
num_val_tasks = len(val_task_ids)
if jax.process_index() == 0:
    print(f'SFT split: train={num_train_tasks}, val={num_val_tasks}, total={env_num_tasks}')
env_task_idx = 0
np_rng = np.random.default_rng(jax.process_index())


def get_sft_batch_from_indices(idxs):
    idxs_local = train_get_local_slice(idxs).tolist()

    token_seqs_local = []
    target_masks_local = []
    for i in idxs_local:
        tokens, target_mask = env.get_sft_data(i)
        tokens = list(tokens)[:seq_len]
        target_mask = list(target_mask)[:seq_len]
        token_seqs_local.append(tokens)
        target_masks_local.append(target_mask)

    token_batch_local, _ = pad_and_collate(token_seqs_local, pad_id, seq_len, 'right', raise_on_truncation=False)
    target_mask_local, _ = pad_and_collate(target_masks_local, 0, seq_len, 'right', raise_on_truncation=False)
    target_mask_local = target_mask_local.astype(np.int32)
    token_batch = host_gather(train_shard_data_fn(token_batch_local))
    target_mask = host_gather(train_shard_data_fn(target_mask_local))
    return np.asarray(token_batch), np.asarray(target_mask)


def get_sft_batch():
    """Builds one global train batch, excluding held-out validation tasks if configured."""
    global env_task_idx
    positions = np.arange(env_task_idx, env_task_idx + global_batch_size) % num_train_tasks
    env_task_idx = (env_task_idx + global_batch_size) % num_train_tasks
    return get_sft_batch_from_indices(train_task_ids[positions])


def get_sft_val_info(train_state):
    if num_val_tasks == 0:
        return {}
    infos = []
    n_batches = max(1, int(config.train.val_batches))
    for b in range(n_batches):
        positions = np.arange(b * global_batch_size, (b + 1) * global_batch_size) % num_val_tasks
        token_batch_local, target_mask_local = get_sft_batch_from_indices(val_task_ids[positions])
        token_batch = shard_microbatched(token_batch_local)
        target_mask = shard_microbatched(target_mask_local)
        info = eval_sft_microbatches(train_state, token_batch, target_mask)
        infos.append(jax.device_get(info))
    info = jax.tree.map(lambda *xs: sum(xs) / len(xs), *infos)
    return {f'val/{k}': float(np.array(v).mean()) for k, v in info.items()}


def shard_microbatched(x):
    x = jnp.reshape(x, (microbatch_size, -1, *x.shape[1:]))
    return train_shard_data_fn(train_get_local_slice(x))


step = 0
train_iter = 0
test_rollouts_rows_by_env = defaultdict(list)
num_test_evals = 0
test_sampler = None

eval_time_interval_secs = parse_wallclock_interval(config.eval_every_wallclock)
max_runtime_secs = parse_wallclock_interval(config.max_runtime) if config.max_runtime else 0
start_time = time.time()
last_eval_time = time.time()
runtime_exceeded = False

while step < config.max_steps and train_iter < config.max_iters and not runtime_exceeded:
    Timer('data').start()
    token_batch_local, target_mask_local = get_sft_batch()
    token_batch = shard_microbatched(token_batch_local)
    target_mask = shard_microbatched(target_mask_local)
    Timer('data').end()

    Timer('update').start()
    train_state, info = update(train_state, token_batch, target_mask)
    Timer('update').end()
    PerformanceMeter.add('flops_train', 3 * flops_per_token * global_batch_size * seq_len)

    info = jax.device_get(info)
    info = jax.tree.map(lambda x: float(np.array(x).mean()), info)
    info['env_epochs'] = (train_iter + 1) * global_batch_size / env_num_tasks if env.num_tasks != -1 else 0
    info.update(prefix_metrics(Timer.times(), 'time/'))
    Timer.reset()

    runtime_exceeded = max_runtime_secs > 0 and time.time() - start_time >= max_runtime_secs
    is_last_step = runtime_exceeded or (step >= config.max_steps - 1) or (train_iter + 1 >= config.max_iters)
    do_log = is_last_step or (config.log_every_steps > 0 and step % config.log_every_steps == 0)
    do_print = is_last_step or (config.print_every_steps > 0 and step % config.print_every_steps == 0)
    do_val = num_val_tasks > 0 and (
        is_last_step or (int(config.train.val_interval) > 0 and step % int(config.train.val_interval) == 0)
    )
    if do_val:
        Timer('val').start()
        info.update(get_sft_val_info(train_state))
        Timer('val').end()
    if do_print:
        print(f'[step {step}] =================== SFT step {step} (Iter {train_iter}) ===================')
        for k, v in sorted(info.items()):
            print(f'[step {step}] {k}: {v}')
    if do_log:
        perf = PerformanceMeter.get()
        perf_cumulative = PerformanceMeter._sum
        PerformanceMeter.take_snapshot()
        log_data = {**info, **{f'perf/{name}': val for name, val in perf.items()}}
        log_data['perf/flops_cumulative'] = sum(v for k, v in perf_cumulative.items() if k.startswith('flops_'))
        if jax.process_index() == 0:
            wandb.log({'global_step': step, 'train_iter': train_iter, **log_data})

    time_since_test = time.time() - last_eval_time
    completed_step = step - 1
    eval_by_step = completed_step > 0 and crossed_multiple(completed_step - 1, completed_step, config.eval_every_steps)
    do_eval = is_last_step or eval_by_step or time_since_test >= eval_time_interval_secs
    if do_eval and test_env is not None:
        test_metrics, test_sampler, sub_rollouts = test_evaluator.eval(
            train_state.params,
            test_env,
            config.test_env.num_epochs,
            sampler=test_sampler,
            verbose=num_test_evals == 0,
        )
        num_test_evals += 1
        last_eval_time = time.time()
        test_info = {f'test_env/{k}': float(np.mean(v)) for k, v in test_metrics.items()}
        for nickname, rows in sub_rollouts.items():
            test_rollouts_rows_by_env[nickname].extend({'step': step, **row} for row in rows)
        if jax.process_index() == 0:
            log_payload = {'global_step': step, 'train_iter': train_iter, **test_info}
            for nickname, rows in test_rollouts_rows_by_env.items():
                if rows:
                    log_payload[f'test_rollouts_table/{nickname}'] = wandb_table_from_rows(rows)
            wandb.log(log_payload)

    step += 1
    train_iter += 1
    if runtime_exceeded:
        print(f'Max runtime of {config.max_runtime} reached. Stopping.')
        break
