### Helpers for sampling from models.
import os
import dataclasses
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from enum import IntEnum

from lmpo.utils.jax_utils import init_jax_compilation_cache, ns
from lmpo.utils.sharding import host_gather, host_local_slice
from lmpo.models.qwen3 import Qwen3Model, KVCache, count_left_padding, length_minus_padding
from lmpo.models.tokenizer import token_ids
from lmpo.utils.array_utils import pad_and_collate
from lmpo.utils.performance_meter import PerformanceMeter
from lmpo.utils.timer import Timer


init_jax_compilation_cache()


FORCING_DISABLED = -(2**30)  # sentinel passed as trigger_step when forcing is off


class TokenRole(IntEnum):
    PAD = 0
    OBSERVATION = 1
    ACTION = 2


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SamplingState:
    cache: KVCache
    tokens: jnp.ndarray  # [B, max_seq_len] int32
    logprobs: jnp.ndarray  # [B, max_seq_len] float32
    role: jnp.ndarray  # [B, max_seq_len] int32 (TokenRole)
    action_idx: jnp.ndarray  # [B, max_seq_len] int32, -1 for non-action
    is_forced: jnp.ndarray  # [B, max_seq_len] bool
    write_pos: jnp.ndarray  # [B] int32, next index to write at
    next_action_idx: jnp.ndarray  # scalar int32

    @classmethod
    def get_sharding(cls, data_shard, tp_size: int = 1):
        no_shard = ns(data_shard.mesh)
        return cls(
            cache=KVCache.get_sharding(data_shard, tp_size=tp_size),
            tokens=data_shard,
            logprobs=data_shard,
            role=data_shard,
            action_idx=data_shard,
            is_forced=data_shard,
            write_pos=data_shard,
            next_action_idx=no_shard,
        )


class Sampler:
    def __init__(self, model: Qwen3Model, tokenizer, data_shard, no_shard, temp: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.data_shard = data_shard
        self.no_shard = no_shard
        self.temp = temp
        self.cache_sharding = KVCache.get_sharding(data_shard, tp_size=model.tp_size)
        self.cache_fns_by_shape = {}
        self.allocate_state_fns_by_shape = {}
        self.decode_scans_by_steps = {}
        self.logprobs_apply = None
        self.prefill_apply = self.compile_prefill_apply()
        self.initialize_cache_from_kv = self.compile_initialize_cache_from_kv()
        self.append_prefill_apply = self.compile_append_prefill_apply()
        self.decode_apply = self.compile_decode_apply()
        self.sample_from_hidden = self.compile_sample_from_hidden()
        ids = token_ids(tokenizer)
        self.token_ids = {
            'end_think_id': ids['</think>'],
            'lt_id': ids['<'],
            'answer_id': ids['answer'],
            'gt_id': ids['>'],
            'newline_id': ids['\n'],
        }

    def compile_cache(self, batch_size, max_seq_len, verbose=False):
        from lmpo.utils.sharding import get_memory_stats

        model = self.model
        cache_sharding = self.cache_sharding
        total_bytes = 4 * model.num_layers * batch_size * max_seq_len * model.kv_heads * model.head_dim
        gib_per_device = total_bytes / jax.device_count() / (1024**3)
        if verbose:
            stats = get_memory_stats()
            gib = 1024**3
            in_use_after = stats['bytes_in_use'] / gib + gib_per_device
            peak_after = max(stats.get('peak_bytes_in_use', stats['bytes_in_use']) / gib, in_use_after)
            limit = (stats.get('bytes_limit') or 0) / gib
            free_after = max(limit - in_use_after, 0.0)
            print(
                f'KVCache size per device: {gib_per_device:.2f} GiB (batch={batch_size}, seq={max_seq_len})\n'
                f'Predicted memory after KVCache: in_use={in_use_after:.2f} peak={peak_after:.2f} '
                f'limit={limit:.2f} free={free_after:.2f} GiB',
                flush=True,
            )

        @partial(jax.jit, out_shardings=cache_sharding)
        def get_cache():
            print(f'JIT compiling get_cache (batch={batch_size}, max_seq_len={max_seq_len})')
            return KVCache.create(model.num_layers, batch_size, max_seq_len, model.head_dim, model.kv_heads)

        return get_cache

    def get_cache(self, batch_size, max_seq_len, verbose=False):
        key = (batch_size, max_seq_len)
        if key not in self.cache_fns_by_shape:
            self.cache_fns_by_shape[key] = self.compile_cache(batch_size, max_seq_len, verbose=verbose)
        return self.cache_fns_by_shape[key]()

    def compile_allocate_state(self, batch_size, max_seq_len, pad_id):
        state_sharding = SamplingState.get_sharding(self.data_shard, tp_size=self.model.tp_size)

        @partial(jax.jit, out_shardings=state_sharding, donate_argnums=(0,))
        def allocate_state(cache):
            print(f'JIT compiling allocate_state (batch={batch_size}, max_seq_len={max_seq_len})')
            return SamplingState(
                cache=cache,
                tokens=jnp.full((batch_size, max_seq_len), pad_id, dtype=jnp.int32),
                logprobs=jnp.zeros((batch_size, max_seq_len), dtype=jnp.float32),
                role=jnp.full((batch_size, max_seq_len), TokenRole.PAD, dtype=jnp.int32),
                action_idx=jnp.full((batch_size, max_seq_len), -1, dtype=jnp.int32),
                is_forced=jnp.zeros((batch_size, max_seq_len), dtype=jnp.bool_),
                write_pos=jnp.zeros((batch_size,), dtype=jnp.int32),
                next_action_idx=jnp.array(0, dtype=jnp.int32),
            )

        return allocate_state

    def allocate_state(self, batch_size, max_seq_len, pad_id, verbose=False):
        cache = self.get_cache(batch_size, max_seq_len, verbose=verbose)
        key = (batch_size, max_seq_len, pad_id)
        if key not in self.allocate_state_fns_by_shape:
            self.allocate_state_fns_by_shape[key] = self.compile_allocate_state(batch_size, max_seq_len, pad_id)
        return self.allocate_state_fns_by_shape[key](cache)

    def compile_prefill_apply(self):
        kv_sharding = [(self.cache_sharding.k, self.cache_sharding.v) for _ in range(self.model.num_layers)]
        model = self.model

        @partial(jax.jit, out_shardings=(self.data_shard, kv_sharding))
        def prefill_apply(params, tokens, token_mask):
            print(f'JIT compiling prefill_apply for tokens of shape {tokens.shape}')
            hidden, kv_list = model.apply(
                {'params': params}, tokens, token_mask, cache=None, return_hidden=True, return_kv=True
            )
            return hidden[:, -1, :], kv_list  # [B, D]

        return prefill_apply

    def compile_initialize_cache_from_kv(self):
        kv_sharding = [(self.cache_sharding.k, self.cache_sharding.v) for _ in range(self.model.num_layers)]

        @partial(
            jax.jit,
            in_shardings=(kv_sharding, self.data_shard, self.cache_sharding),
            out_shardings=self.cache_sharding,
            donate_argnums=(2,),
        )
        def initialize_cache_from_kv(kv_list, token_mask, cache):
            print(f'JIT compiling initialize_cache_from_kv for token_mask of shape {token_mask.shape}')
            for layer_id, (k, v) in enumerate(kv_list):
                cache.k[layer_id] = jax.lax.dynamic_update_slice(cache.k[layer_id], k, (0, 0, 0, 0))
                cache.v[layer_id] = jax.lax.dynamic_update_slice(cache.v[layer_id], v, (0, 0, 0, 0))
            inc = length_minus_padding(token_mask)
            return cache.replace(lengths=cache.lengths + inc)

        return initialize_cache_from_kv

    def compile_append_prefill_apply(self):
        model = self.model

        @partial(jax.jit, out_shardings=(self.data_shard, self.cache_sharding), donate_argnums=(3,))
        def append_prefill_apply(params, tokens, token_mask, cache):
            print(f'JIT compiling append_prefill_apply for tokens of shape {tokens.shape}')
            hidden, cache = model.apply({'params': params}, tokens, token_mask, cache=cache, return_hidden=True)
            last_idx = jnp.maximum(length_minus_padding(token_mask) - 1, 0).astype(jnp.int32)
            hidden_idx = last_idx[:, None, None].repeat(hidden.shape[-1], axis=-1)
            hidden = jnp.take_along_axis(hidden, hidden_idx, axis=1)[:, 0, :]
            return hidden, cache  # [B, D]

        return append_prefill_apply

    def compile_decode_apply(self):
        model = self.model

        @partial(jax.jit, out_shardings=(self.data_shard, self.cache_sharding), donate_argnums=(3,))
        def decode_apply(params, tokens, token_mask, cache):
            print(f'JIT compiling decode_apply for tokens of shape {tokens.shape}')
            hidden, cache = model.apply({'params': params}, tokens, token_mask, cache=cache, return_hidden=True)
            return hidden[:, 0, :], cache  # [B, D]

        return decode_apply

    def compile_sample_from_hidden(self):
        hidden_sharding = ns(self.data_shard.mesh, 'data', None)
        data_shard = self.data_shard
        temp = self.temp

        @partial(jax.jit, in_shardings=(hidden_sharding, None, None), out_shardings=(data_shard, data_shard))
        def sample_from_hidden(hidden, params, rng):
            print(f'JIT compiling sample_from_hidden for hidden of shape {hidden.shape}, temp={temp}')
            logits = (hidden @ params['Dense_0']['kernel']).astype(jnp.float32)  # [B, vocab]
            if temp == 0:
                sampled_token = jnp.argmax(logits, axis=-1)
            else:
                sampled_token = jax.random.categorical(rng, logits / temp, axis=-1)
            logprobs = jax.nn.log_softmax(logits / (temp if temp != 0 else 1), axis=-1)
            sampled_logprob = jnp.take_along_axis(logprobs, sampled_token[:, None], axis=-1)[:, 0]
            return sampled_token, sampled_logprob

        return sample_from_hidden

    def decode_scan_for_steps(self, num_steps: int):
        if num_steps in self.decode_scans_by_steps:
            return self.decode_scans_by_steps[num_steps]

        decode_apply = self.decode_apply
        sample_from_hidden = self.sample_from_hidden
        data_shard = self.data_shard
        cache_sharding = self.cache_sharding
        no_shard = self.no_shard
        end_think_id = self.token_ids['end_think_id']
        lt_id = self.token_ids['lt_id']
        answer_id = self.token_ids['answer_id']
        gt_id = self.token_ids['gt_id']
        newline_id = self.token_ids['newline_id']
        answer_force_seq = (newline_id, newline_id, lt_id, answer_id, gt_id)

        def update_answer_hdr_state(state, token):
            is_lt = token == lt_id
            is_answer = token == answer_id
            is_gt = token == gt_id
            s0, s1, s2, s3 = state == 0, state == 1, state == 2, state == 3
            next_state = state
            next_state = jnp.where(s0 & is_lt, 1, next_state)
            next_state = jnp.where(s1 & is_answer, 2, next_state)
            next_state = jnp.where(s1 & ~is_answer & is_lt, 1, next_state)
            next_state = jnp.where(s1 & ~is_answer & ~is_lt, 0, next_state)
            next_state = jnp.where(s2 & is_gt, 3, next_state)
            next_state = jnp.where(s2 & ~is_gt & is_lt, 1, next_state)
            next_state = jnp.where(s2 & ~is_gt & ~is_lt, 0, next_state)
            next_state = jnp.where(s3, 3, next_state)
            return next_state

        def step_body(
            params,
            rng,
            cache,
            sampled_token,
            saw_end_think,
            answer_hdr_state,
            step_idx,
            end_think_trigger_step,
            answer_trigger_step,
        ):
            next_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
            hidden, cache = decode_apply(params, sampled_token[:, None], next_token_mask[:, None], cache)
            sampled_token, sampled_logprobs = sample_from_hidden(hidden, params, rng)

            forced = jnp.zeros(sampled_token.shape, dtype=jnp.bool_)

            should_force = ~saw_end_think & (step_idx == end_think_trigger_step)
            sampled_token = jnp.where(should_force, end_think_id, sampled_token)
            sampled_logprobs = jnp.where(should_force, 0.0, sampled_logprobs)
            forced = forced | should_force

            for offset, forced_id in enumerate(answer_force_seq):
                should_force = (answer_hdr_state != 3) & (step_idx == answer_trigger_step + offset)
                sampled_token = jnp.where(should_force, forced_id, sampled_token)
                sampled_logprobs = jnp.where(should_force, 0.0, sampled_logprobs)
                forced = forced | should_force

            saw_end_think = saw_end_think | (sampled_token == end_think_id)
            answer_hdr_state = update_answer_hdr_state(answer_hdr_state, sampled_token)

            return sampled_token, sampled_logprobs, forced, saw_end_think, answer_hdr_state, cache

        @partial(
            jax.jit,
            in_shardings=(None, None, cache_sharding, data_shard, data_shard, data_shard, no_shard, no_shard),
            out_shardings=(data_shard, data_shard, data_shard, cache_sharding),
            donate_argnums=(2,),
        )
        def decode_scan(
            params,
            rng,
            cache,
            sampled_token,
            saw_end_think,
            answer_hdr_state,
            end_think_trigger_step,
            answer_trigger_step,
        ):
            print(f'JIT compiling decode_scan for {num_steps} steps')

            def body(carry, _):
                rng, cache, sampled_token, saw_end_think, answer_hdr_state, step_idx = carry
                key, rng = jax.random.split(rng)
                sampled_token, sampled_logprobs, forced, saw_end_think, answer_hdr_state, cache = step_body(
                    params,
                    key,
                    cache,
                    sampled_token,
                    saw_end_think,
                    answer_hdr_state,
                    step_idx,
                    end_think_trigger_step,
                    answer_trigger_step,
                )
                carry = (rng, cache, sampled_token, saw_end_think, answer_hdr_state, step_idx + 1)
                return carry, (sampled_token, sampled_logprobs, forced)

            step_idx = jnp.zeros((), dtype=jnp.int32)
            carry = (rng, cache, sampled_token, saw_end_think, answer_hdr_state, step_idx)
            carry, (tokens, logprobs, is_forced) = jax.lax.scan(body, carry, xs=None, length=num_steps)
            cache = carry[1]
            return tokens.T, logprobs.T, is_forced.T, cache  # [batch, time]

        self.decode_scans_by_steps[num_steps] = decode_scan
        return decode_scan

    def compile_logprobs_apply(self):
        flash_model = dataclasses.replace(self.model, use_flash_attn=True)

        @partial(jax.jit, out_shardings=self.data_shard)
        def logprobs_apply(params, tokens, token_mask, target_tokens, target_mask):
            print(f'JIT compiling logprobs_apply for tokens of shape {tokens.shape}')
            logits, _ = flash_model.apply({'params': params}, tokens, token_mask, cache=None)
            logits = logits.astype(jnp.float32)
            log_norm = jax.nn.logsumexp(logits, axis=-1)
            token_logprobs = jnp.take_along_axis(logits, target_tokens[..., None], axis=-1)[..., 0] - log_norm
            return jnp.where(target_mask, token_logprobs, 0.0)

        return logprobs_apply

    def sample(
        self,
        params,
        prompt_tokens,
        num_generation_tokens,
        rng,
        state: SamplingState = None,
        max_seq_len: int = None,
        force_answer_at=-1,
        force_end_think_at=-1,
        verbose=True,
        profile_dir='',
        timer_name='',
    ):
        """
        Samples tokens autoregressively, and can batch for performance.
        Args:
            prompt_tokens: An array of tokens, padded by `pad_id` on the LEFT. [batch, time].
            force_answer_at: If > 0, forces the insertion of an <answer> tag at (force_answer_at) tokens before the end of the generation.
        """
        pad_id = self.tokenizer.pad_token_id
        batch_size = prompt_tokens.shape[0]
        token_mask = jnp.where(prompt_tokens != pad_id, 1, 0).astype(jnp.int32)
        if num_generation_tokens <= 0:
            raise ValueError(f'num_generation_tokens must be positive, got {num_generation_tokens}.')

        if state is None:
            if max_seq_len is None:
                raise ValueError('max_seq_len must be provided when state is None.')
            state = self.allocate_state(batch_size, max_seq_len, pad_id, verbose=verbose)
            state = dataclasses.replace(
                state,
                cache=state.cache.replace(starts=count_left_padding(prompt_tokens, pad_id=pad_id)),
            )
            append_to_cache = False
        else:
            append_to_cache = True
            if max_seq_len is not None and max_seq_len != state.tokens.shape[1]:
                raise ValueError(
                    f'max_seq_len={max_seq_len} does not match existing state length {state.tokens.shape[1]}.'
                )

        state_prompt_tokens, state_prompt_mask, prompt_span = compact_left_padded_tokens(
            prompt_tokens, token_mask, pad_id
        )
        prefill_tokens = state_prompt_tokens if append_to_cache else prompt_tokens
        prefill_mask = state_prompt_mask if append_to_cache else token_mask

        prefill_profile_dir = os.path.join(profile_dir, 'prefill') if profile_dir else ''
        decode_profile_dir = os.path.join(profile_dir, 'decode') if profile_dir else ''

        key, rng = jax.random.split(rng)
        if prefill_profile_dir and jax.process_index() == 0:
            os.makedirs(prefill_profile_dir, exist_ok=True)
            jax.profiler.start_trace(prefill_profile_dir)
        if timer_name:
            Timer(f'{timer_name}/prefill').start()
        prev_write_pos = state.write_pos
        if append_to_cache:
            hidden, cache = self.append_prefill_apply(params, prefill_tokens, prefill_mask, state.cache)
        else:
            hidden, kv_list = self.prefill_apply(params, prefill_tokens, prefill_mask)
            cache = self.initialize_cache_from_kv(kv_list, prefill_mask, state.cache)

        prompt_logprobs = jnp.zeros(prompt_tokens.shape, dtype=jnp.float32)
        prompt_role = jnp.where(state_prompt_mask != 0, TokenRole.OBSERVATION, TokenRole.PAD).astype(jnp.int32)
        prompt_action_idx = jnp.full(prompt_tokens.shape, -1, dtype=jnp.int32)
        prompt_is_forced = jnp.zeros(prompt_tokens.shape, dtype=jnp.bool_)
        state = dataclasses.replace(
            state,
            cache=cache,
            tokens=dynamic_update_rows(state.tokens, state_prompt_tokens.astype(jnp.int32), prev_write_pos),
            logprobs=dynamic_update_rows(state.logprobs, prompt_logprobs, prev_write_pos),
            role=dynamic_update_rows(state.role, prompt_role, prev_write_pos),
            action_idx=dynamic_update_rows(state.action_idx, prompt_action_idx, prev_write_pos),
            is_forced=dynamic_update_rows(state.is_forced, prompt_is_forced, prev_write_pos),
            write_pos=prev_write_pos + prompt_span,
        )
        sampled_token, sampled_logprobs = self.sample_from_hidden(hidden, params, key)
        _ = jax.device_get(host_gather(sampled_token))
        if timer_name:
            Timer(f'{timer_name}/prefill').end()
        if prefill_profile_dir:
            if jax.process_index() == 0:
                jax.profiler.stop_trace()
            print(f'[process {jax.process_index()}] prefill profile saved to {prefill_profile_dir}')

        first_is_forced = jax.device_put(jnp.zeros(batch_size, dtype=jnp.bool_), sampled_token.sharding)
        end_think_trigger_step = jax.device_put(
            jnp.array(
                num_generation_tokens - force_end_think_at if force_end_think_at > 0 else FORCING_DISABLED,
                dtype=jnp.int32,
            ),
            self.no_shard,
        )
        answer_trigger_step = jax.device_put(
            jnp.array(
                num_generation_tokens - force_answer_at if force_answer_at > 0 else FORCING_DISABLED,
                dtype=jnp.int32,
            ),
            self.no_shard,
        )

        saw_end_think = jax.device_put(jnp.zeros(batch_size, dtype=jnp.bool_), self.data_shard)
        answer_hdr_state = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), self.data_shard)

        if timer_name:
            Timer(f'{timer_name}/generate').start()
        if decode_profile_dir and jax.process_index() == 0:
            os.makedirs(decode_profile_dir, exist_ok=True)
            jax.profiler.start_trace(decode_profile_dir)
        gen_tokens, gen_logprobs, gen_forced, cache = self.decode_scan_for_steps(num_generation_tokens - 1)(
            params,
            rng,
            cache,
            sampled_token,
            saw_end_think,
            answer_hdr_state,
            end_think_trigger_step,
            answer_trigger_step,
        )
        if decode_profile_dir:
            jax.block_until_ready(gen_tokens)
            if jax.process_index() == 0:
                jax.profiler.stop_trace()
            print(f'[process {jax.process_index()}] decode profile saved to {decode_profile_dir}')
        tokens = jnp.concatenate([sampled_token.astype(jnp.int32)[:, None], gen_tokens], axis=-1)  # [batch, time]
        logprobs = jnp.concatenate([sampled_logprobs[:, None], gen_logprobs], axis=-1)  # [batch, time]
        is_forced = jnp.concatenate([first_is_forced[:, None], gen_forced], axis=-1)  # [batch, time]

        sampled_token = tokens[:, -1]
        final_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
        _, cache = self.decode_apply(params, sampled_token[:, None], final_token_mask[:, None], cache)
        state = dataclasses.replace(state, cache=cache)
        action_cache_start = cache.lengths - jnp.array(num_generation_tokens, dtype=jnp.int32)

        eos_id = self.tokenizer.eos_token_id
        has_eos = jnp.any(tokens == eos_id, axis=-1)
        eos_idx = jnp.argmax(tokens == eos_id, axis=-1).astype(jnp.int32)
        action_len = jnp.where(has_eos, eos_idx + 1, num_generation_tokens).astype(jnp.int32)
        action_time = jnp.arange(num_generation_tokens, dtype=jnp.int32)[None, :]
        action_mask = action_time < action_len[:, None]
        tokens = jnp.where(action_mask, tokens, pad_id)
        logprobs = jnp.where(action_mask, logprobs, 0.0)
        is_forced = jnp.where(action_mask, is_forced, False)
        action_role = jnp.where(action_mask, TokenRole.ACTION, TokenRole.PAD).astype(jnp.int32)
        action_idx = jnp.where(action_mask, state.next_action_idx, -1).astype(jnp.int32)
        cache = cache.replace(lengths=action_cache_start + action_len)
        state = dataclasses.replace(
            state,
            cache=cache,
            tokens=dynamic_update_rows(state.tokens, tokens.astype(jnp.int32), state.write_pos),
            logprobs=dynamic_update_rows(state.logprobs, logprobs, state.write_pos),
            role=dynamic_update_rows(state.role, action_role, state.write_pos),
            action_idx=dynamic_update_rows(state.action_idx, action_idx, state.write_pos),
            is_forced=dynamic_update_rows(state.is_forced, is_forced, state.write_pos),
            write_pos=state.write_pos + action_len,
            next_action_idx=state.next_action_idx + jnp.array(1, dtype=jnp.int32),
        )
        _ = jax.device_get(host_gather(logprobs))
        _ = jax.device_get(host_gather(is_forced))
        if timer_name:
            Timer(f'{timer_name}/generate').end()
            prefill_elapsed = Timer.get_last(f'{timer_name}/prefill')
            generate_elapsed = Timer.get_last(f'{timer_name}/generate')
            tokens_in = int(np.prod(prompt_tokens.shape))
            tokens_out = int(np.prod(tokens.shape))
            PerformanceMeter.average(f'{timer_name}/generate_output_tokens_per_sec', tokens_out / generate_elapsed)
            PerformanceMeter.average(
                f'{timer_name}/end_to_end_tokens_per_sec',
                (tokens_in + tokens_out) / (prefill_elapsed + generate_elapsed),
            )
        return state

    def get_local_action_tokens(self, state, action_idx, *, include_logprobs=False):
        """Extract this process's action rows without an interhost gather."""
        tokens_local = host_local_slice(state.tokens)
        roles_local = host_local_slice(state.role)
        action_idx_local = host_local_slice(state.action_idx)
        masks = (roles_local == int(TokenRole.ACTION)) & (action_idx_local == action_idx)
        action_tokens = [row[mask].tolist() for row, mask in zip(tokens_local, masks)]

        if not include_logprobs:
            return action_tokens

        logprobs_local = host_local_slice(state.logprobs)
        action_logprobs = [row[mask].tolist() for row, mask in zip(logprobs_local, masks)]
        return action_tokens, action_logprobs

    def logprobs_for_tokens(self, params, full_tokens, target_mask, pad_id=None):
        """Per-token log-probs for positions where target_mask is True."""
        pad_id = self.tokenizer.pad_token_id if pad_id is None else pad_id
        if self.logprobs_apply is None:
            self.logprobs_apply = self.compile_logprobs_apply()
        token_mask = jnp.where(full_tokens != pad_id, 1, 0).astype(jnp.int32)
        target_tokens = jnp.concatenate([full_tokens[:, 1:], jnp.zeros_like(full_tokens[:, :1])], axis=1)
        target_mask_shifted = jnp.concatenate([target_mask[:, 1:], jnp.zeros_like(target_mask[:, :1])], axis=1)
        return host_gather(self.logprobs_apply(params, full_tokens, token_mask, target_tokens, target_mask_shifted))


def dynamic_update_rows(base, update, starts):
    return jax.vmap(lambda row, upd, start: jax.lax.dynamic_update_slice_in_dim(row, upd, start, axis=0))(
        base, update, starts
    )


def compact_left_padded_tokens(tokens, mask, pad_id):
    left_pad = count_left_padding(tokens, pad_id=pad_id).astype(jnp.int32)
    nonpad_len = jnp.sum(mask != 0, axis=-1).astype(jnp.int32)
    time = jnp.arange(tokens.shape[1], dtype=jnp.int32)
    src_idx = left_pad[:, None] + time[None, :]
    src_idx_clipped = jnp.minimum(src_idx, tokens.shape[1] - 1)
    compact_tokens = jnp.take_along_axis(tokens, src_idx_clipped, axis=1)
    compact_tokens = jnp.where(time[None, :] < nonpad_len[:, None], compact_tokens, pad_id)
    compact_mask = (time[None, :] < nonpad_len[:, None]).astype(jnp.int32)
    return compact_tokens, compact_mask, nonpad_len


if __name__ == '__main__':
    import argparse

    from lmpo.models.qwen3 import (
        create_model_from_ckpt,
        get_tp_spec,
    )
    from lmpo.utils.sharding import create_sharding, get_shard_params_fn
    from lmpo.models.tokenizer import create_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    model_dir = parser.parse_args().model_dir
    if model_dir.startswith('model_dir='):
        model_dir = model_dir.split('=', 1)[1]
    fsdp, tp_size, inference_batch_per_device = False, 1, 1
    num_generation_tokens = followup_generation_tokens = 64
    model_kwargs = dict(use_flash_attn=True, mesh=None, tp_size=tp_size)
    model, params_source = create_model_from_ckpt(model_dir, **model_kwargs)
    param_shard, no_shard, data_shard, shard_data_fn, _ = create_sharding(
        params_source, fsdp=fsdp, tp_size=tp_size, get_tp_spec=get_tp_spec
    )
    sampling_model = dataclasses.replace(model, mesh=data_shard.mesh)
    params = get_shard_params_fn(param_shard, no_shard)(params_source)
    tokenizer = create_tokenizer(model_dir)
    sampler = Sampler(sampling_model, tokenizer, data_shard, no_shard)
    assert jax.local_device_count() % tp_size == 0, (
        f'local_device_count={jax.local_device_count()} not divisible by {tp_size=}'
    )
    local_batch_size = jax.local_device_count() // tp_size * inference_batch_per_device

    pad_id = tokenizer.pad_token_id

    def chat_batch(texts):
        token_lists = [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}], add_generation_prompt=True, enable_thinking=False
            )
            for text in texts
        ]
        tokens, _ = pad_and_collate(token_lists, pad_id=pad_id, force_length=128)
        return shard_data_fn(tokens)

    labels = ['cat', 'dog', 'bird', 'fish', 'elephant', 'tiger', 'lion', 'giraffe', 'zebra', 'monkey']
    poem_prompts = [f'Write a haiku about a {labels[np.random.randint(len(labels))]}.' for _ in range(local_batch_size)]
    first_prompt_tokens = chat_batch(poem_prompts)
    followup_batch = chat_batch(['Now write one sentence reflecting on the poem you just wrote.'] * local_batch_size)
    max_seq_len = (
        first_prompt_tokens.shape[1] + num_generation_tokens + followup_batch.shape[1] + followup_generation_tokens
    )

    print('Multiturn sampling...')
    rng = jax.random.PRNGKey(0)
    state = sampler.sample(
        params,
        first_prompt_tokens,
        num_generation_tokens,
        rng,
        max_seq_len=max_seq_len,
        timer_name='multiturn_t1',
    )

    rng, key = jax.random.split(rng)
    state = sampler.sample(
        params,
        followup_batch,
        followup_generation_tokens,
        key,
        state=state,
        timer_name='multiturn_t2',
    )
    full_state_tokens = np.asarray(host_gather(state.tokens))
    action_idx = np.asarray(host_gather(state.action_idx))
    responses = [tokenizer.decode(row[mask]) for row, mask in zip(full_state_tokens, action_idx == 0)]
    followup_responses = [tokenizer.decode(row[mask]) for row, mask in zip(full_state_tokens, action_idx == 1)]
    for i, text in enumerate(poem_prompts):
        print(f' ======= {text} =======')
        print(responses[i].split('<|im_end|>')[0])
        print('--- Followup ---')
        print(followup_responses[i].split('<|im_end|>')[0])

    print('========= Full raw decoded tokens =========')
    print(tokenizer.decode(full_state_tokens[0].tolist()))
    print('Total tokens shape', full_state_tokens.shape)
    print('========= Timer ===========')
    for name, secs in sorted(Timer.times().items()):
        print(f'  {name}: {secs:.3f}s')
    print('========= PerformanceMeter ========')
    for name, val in sorted(PerformanceMeter.get().items()):
        print(f'  {name}: {val:.3f}')
    print('===================================')
