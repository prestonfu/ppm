"""
Model structure for Qwen3. Taken from https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py.
"""

import json
import glob
import inspect
import os
import numpy as np
import dataclasses
from functools import partial
from safetensors import safe_open
import re
from jax.sharding import Mesh

import jax
import jax.numpy as jnp
import flax

PLATFORM = jax.devices()[0].platform  # 'tpu', 'gpu', or 'cpu'
if PLATFORM == 'tpu':
    from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, SegmentIds
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as pallas_flash_attention_tpu,
    )

import flax.linen as nn
import einops

from lmpo.utils.jax_utils import P, ns
from lmpo.utils.sharding import get_mesh
from lmpo.utils.checkpoint import Checkpoint


def rms_norm(x, gamma, eps):
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def count_left_padding(ids, pad_id=0):
    return jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1)


def length_minus_padding(token_mask):
    return jnp.sum(jnp.cumsum(jnp.flip(token_mask != 0, -1), axis=-1) > 0, -1)


def get_positions(token_mask):
    """Counts positions for segment ids."""

    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(token_mask), token_mask)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype='int32')


def generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: float,
) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        'BT,k->BTk',
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def repeat_kv_heads(k, v, num_reps: int):
    return (
        einops.repeat(k, 'b s h d -> b s (h r) d', r=num_reps),
        einops.repeat(v, 'b s h d -> b s (h r) d', r=num_reps),
    )


def _flash_attention_inner(q, k, v, token_mask):
    if PLATFORM == 'gpu':
        is_prefill = k.shape[1] == q.shape[1]
        mask = (token_mask[:, None, None, :] != 0) if is_prefill else None
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / q.shape[-1] ** 0.5,
            is_causal=is_prefill,
            mask=mask,
        ).astype(jnp.bfloat16)

    blocksize_q = 128
    blocksize_k = 128
    k, v = repeat_kv_heads(k, v, q.shape[2] // k.shape[2])
    query_length = q.shape[1]
    value_length = v.shape[1]
    block_sizes = BlockSizes(
        block_q=min(blocksize_q, query_length),
        block_k_major=min(blocksize_k, value_length),
        block_k=min(blocksize_k, value_length),
        block_b=1,
        block_q_major_dkv=min(blocksize_q, query_length),
        block_k_major_dkv=min(blocksize_k, value_length),
        block_k_dkv=min(blocksize_k, value_length),
        block_q_dkv=min(blocksize_q, query_length),
        block_k_major_dq=min(blocksize_k, value_length),
        block_k_dq=min(blocksize_k, value_length),
        block_q_dq=min(blocksize_q, query_length),
    )

    segment_ids = SegmentIds(token_mask, token_mask)

    return (
        pallas_flash_attention_tpu(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            sm_scale=1.0 / q.shape[-1] ** 0.5,
            block_sizes=block_sizes,
            causal=True if k.shape[1] == q.shape[1] else False,
            segment_ids=segment_ids,
        )
        .transpose(0, 2, 1, 3)
        .astype(jnp.bfloat16)
    )


def _make_flash_attention(mesh: Mesh, tp_size: int):
    if tp_size > 1:
        qkv, mask = P('data', None, 'model', None), P('data', None)
    else:
        qkv, mask = P('data', None, None, None), P('data', None)
    kw = {'mesh': mesh, 'in_specs': (qkv, qkv, qkv, mask), 'out_specs': qkv}
    if 'check_vma' in inspect.signature(jax.shard_map).parameters:
        kw['check_vma'] = False

    @jax.remat
    @partial(jax.shard_map, **kw)
    def flash_sharded(q, k, v, token_mask):
        return _flash_attention_inner(q, k, v, token_mask)

    return flash_sharded


_flash_sharded_cache: dict = {}


def get_flash_sharded_apply(mesh: Mesh | None, tp_size: int):
    m = mesh if mesh is not None else get_mesh(jax.device_count(), 1)
    key = (id(m), tp_size)
    if key not in _flash_sharded_cache:
        _flash_sharded_cache[key] = _make_flash_attention(m, tp_size)
    return _flash_sharded_cache[key]


class KVCache(flax.struct.PyTreeNode):
    k: list[jax.Array]
    v: list[jax.Array]
    lengths: jax.Array
    starts: jax.Array

    @classmethod
    def create(cls, num_layers, batch_size, max_seq_len, head_dim, kv_heads):
        k = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        v = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
        starts = jnp.zeros((batch_size,), dtype=jnp.int32)
        return cls(k=k, v=v, lengths=lengths, starts=starts)

    @classmethod
    def get_sharding(cls, data_shard, tp_size: int = 1):
        if tp_size > 1:
            mesh = data_shard.mesh
            kv_shard = ns(mesh, 'data', None, 'model', None)
            meta_shard = ns(mesh, 'data')
        else:
            kv_shard = data_shard
            meta_shard = data_shard
        return KVCache(k=kv_shard, v=kv_shard, lengths=meta_shard, starts=meta_shard)


class Block(nn.Module):
    """A standard transformer block. Has residual connection, self-attention, and a two-layer MLP."""

    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    mlp_ffw_size: int
    mesh: Mesh | None = None
    tp_size: int = 1
    eps: float = 1e-6
    use_flash_attn: bool = True

    @nn.compact
    def __call__(self, x, sin, cos, mask, token_mask, layer_id, cache=None):
        if self.tp_size > 1:
            NS = partial(ns, self.mesh)
            sc_qkv4d = partial(jax.lax.with_sharding_constraint, shardings=NS('data', None, 'model', None))
            sc_qkv3d = partial(jax.lax.with_sharding_constraint, shardings=NS('data', None, 'model'))
            sc_hidden = partial(jax.lax.with_sharding_constraint, shardings=NS('data', None, None))

        # =========================
        # === Self-Attention Block.
        # =========================

        pre_gamma = self.param('pre_gamma', nn.initializers.constant(1.0), (self.hidden_size,))
        q_gamma = self.param('q_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        k_gamma = self.param('k_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        x_norm = rms_norm(x, pre_gamma, self.eps)

        # Calculate Q,K,V via fused projection.
        q_dim = self.q_heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        qkv_raw = nn.Dense(q_dim + 2 * kv_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        q, k, v = jnp.split(qkv_raw, [q_dim, q_dim + kv_dim], axis=-1)
        q = jnp.reshape(q, (q.shape[0], q.shape[1], self.q_heads, self.head_dim))
        k = jnp.reshape(k, (k.shape[0], k.shape[1], self.kv_heads, self.head_dim))
        v = jnp.reshape(v, (v.shape[0], v.shape[1], self.kv_heads, self.head_dim))

        q = rms_norm(q, q_gamma, self.eps)
        q = apply_rotary_embedding(q, sin, cos)
        k = rms_norm(k, k_gamma, self.eps)
        k = apply_rotary_embedding(k, sin, cos)

        if cache is not None:

            def _update_row(cache_row, new_row, start):
                return jax.lax.dynamic_update_slice_in_dim(cache_row, new_row, start, axis=0)

            k = jax.vmap(_update_row)(cache.k[layer_id], k, cache.lengths)
            v = jax.vmap(_update_row)(cache.v[layer_id], v, cache.lengths)

        if self.tp_size > 1:
            q, k, v = sc_qkv4d(q), sc_qkv4d(k), sc_qkv4d(v)

        flash_eligible = cache is None and self.use_flash_attn
        if flash_eligible:
            qkv = get_flash_sharded_apply(self.mesh, self.tp_size)(q, k, v, token_mask)
            qkv = jnp.reshape(qkv, (qkv.shape[0], qkv.shape[1], self.q_heads * self.head_dim))
        else:
            b, t, qh, d = q.shape
            _, T, kh, _ = k.shape
            q = jnp.reshape(q, (b, t, kh, qh // kh, d))
            qk = jnp.einsum('bthgd,bThd->btThg', q, k) * (d**-0.5)
            qk = jnp.reshape(qk, (b, t, T, qh))
            qk = jnp.where(mask, qk, -1e30)
            attn = jax.nn.softmax(qk.astype(jnp.float32), axis=2)
            attn = jnp.reshape(attn, (b, t, T, kh, qh // kh))
            qkv = jnp.einsum('btThg,bThd->bthgd', attn, v).astype(x.dtype)
            qkv = jnp.reshape(qkv, (b, t, qh * d))

        if self.tp_size > 1:
            qkv = sc_qkv3d(qkv)

        attn_x = nn.Dense(self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(qkv)
        if self.tp_size > 1:
            attn_x = sc_hidden(attn_x)

        x = x + attn_x

        # =========================
        # === MLP Block.
        # =========================
        post_gamma = self.param('post_gamma', nn.initializers.constant(1.0), (self.hidden_size,))
        x_norm = rms_norm(x, post_gamma, self.eps)
        g = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        g = nn.silu(g)
        y = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        if self.tp_size > 1:
            g, y = sc_qkv3d(g), sc_qkv3d(y)
        y = g * y
        mlp_x = nn.Dense(features=self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(y)
        if self.tp_size > 1:
            mlp_x = sc_hidden(mlp_x)
        x = x + mlp_x
        return x, k, v


class Qwen3Model(nn.Module):
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    mlp_ffw_size: int
    num_layers: int
    rope_theta: int
    mesh: Mesh | None = None
    tp_size: int = 1
    eps: float = 1e-6
    use_remat: bool = False
    use_flash_attn: bool = True
    use_v_head: bool = False

    @nn.compact
    def __call__(self, x, token_mask, cache=None, get_logits=True, return_hidden=False, return_kv=False):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, dtype=jnp.bfloat16)(x)
        token_mask = token_mask.astype(jnp.int32)
        positions = get_positions(token_mask)
        if cache is not None:
            start_indices = jnp.where(cache.lengths != 0, cache.lengths - cache.starts, 0)
        else:
            start_indices = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        positions = start_indices[:, None] + positions
        sin, cos = generate_pos_embeddings(positions, self.head_dim, self.rope_theta)
        sin, cos = sin.astype(jnp.bfloat16), cos.astype(jnp.bfloat16)

        flash_eligible = cache is None and self.use_flash_attn

        # Attention Mask: compute once here, reuse across layers. Flash attention handles it internally.
        if flash_eligible:
            mask = None
        else:
            if cache is not None:
                T = cache.k[0].shape[1]
                time_idx = jnp.arange(0, T, dtype=jnp.int32)[None, :]  # [1, seqlen]
                q_idx = jnp.where(token_mask != 0, 1, 0)  # [B, seqlen] where tokens exist.
                incremental_pos = length_minus_padding(token_mask)  # [B]
                k_idx = (time_idx >= cache.starts[:, None]) & (
                    time_idx < (cache.lengths + incremental_pos)[:, None].astype(jnp.int32)
                )
                q_offset = cache.lengths  # [B]
            else:
                T = x.shape[1]
                q_idx, k_idx = token_mask, token_mask
                q_offset = jnp.zeros((x.shape[0],), dtype=jnp.int32)

            mask = q_idx[:, :, None] & k_idx[:, None, :]
            mask = mask[:, None, :, :]  # [B, 1, t, T]
            qk_size = (x.shape[0], 1, x.shape[1], T)
            q_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 2)
            k_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 3)
            q_positions = q_iota + q_offset[:, None, None, None]
            causal_mask = q_positions >= k_iota
            mask = jnp.logical_and(mask, causal_mask)
            mask = jnp.transpose(mask, (0, 2, 3, 1))  # [B, t, T, 1]

        BlockFn = Block if not self.use_remat else nn.remat(Block, static_argnums=(6,))
        kv_list = []
        for layer_id in range(self.num_layers):
            x, k, v = BlockFn(
                hidden_size=self.hidden_size,
                q_heads=self.q_heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                mlp_ffw_size=self.mlp_ffw_size,
                mesh=self.mesh,
                tp_size=self.tp_size,
                eps=self.eps,
                use_flash_attn=self.use_flash_attn,
            )(x, sin, cos, mask, token_mask, layer_id, cache)
            if cache is not None:
                cache.k[layer_id] = k
                cache.v[layer_id] = v
            kv_list.append((k, v))

        gamma_final = self.param('gamma_final', nn.initializers.constant(1.0), (self.hidden_size,))
        x = rms_norm(x, gamma_final, self.eps)

        if cache is not None:
            cache = cache.replace(lengths=cache.lengths + length_minus_padding(token_mask))

        if return_hidden:
            if return_kv:
                return x, kv_list
            return x, cache

        if not self.use_v_head:
            if get_logits:
                logits = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.bfloat16)(x)
                if self.tp_size > 1 and self.mesh is not None:
                    logits = jax.lax.with_sharding_constraint(logits, ns(self.mesh, 'data', None, 'model'))
            else:
                logits = None
        else:
            logits = nn.Dense(1, use_bias=False, dtype=jnp.bfloat16)(x)

        return logits, cache


###############################
##### Utils for loading models.
###############################


def create_model_from_hf(hf_dir: str):
    with open(os.path.join(hf_dir, 'config.json')) as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta'],
        use_remat=False,
        use_flash_attn=False,
    )
    tokens = jnp.ones((1, 1), dtype=jnp.int32)
    idx = jnp.ones((1, 1), dtype=jnp.int32)
    params = jax.eval_shape(model.init, jax.random.PRNGKey(0), tokens, idx)['params']

    # Q/K/V are loaded separately then fused into Dense_0; everything else maps directly.
    _HF_KEY_MAPPING = {
        r'model\.embed_tokens\.weight': 'Embed_0.embedding',
        r'model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight': r'Block_\1.Dense_1.kernel',
        # norms
        r'model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight': r'Block_\1.q_gamma',
        r'model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight': r'Block_\1.k_gamma',
        # layer norms (pre/post attention)
        r'model\.layers\.([0-9]+)\.input_layernorm\.weight': r'Block_\1.pre_gamma',
        r'model\.layers\.([0-9]+)\.post_attention_layernorm\.weight': r'Block_\1.post_gamma',
        # mlp (fused layout: gate=Dense_2, up=Dense_3, down=Dense_4)
        r'model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight': r'Block_\1.Dense_2.kernel',
        r'model\.layers\.([0-9]+)\.mlp\.up_proj\.weight': r'Block_\1.Dense_3.kernel',
        r'model\.layers\.([0-9]+)\.mlp\.down_proj\.weight': r'Block_\1.Dense_4.kernel',
        r'model\.norm\.weight': 'gamma_final',
        r'lm_head\.weight': 'Dense_0.kernel',
    }
    _QKV_PATTERNS = {
        r'model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight': 'q',
        r'model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight': 'k',
        r'model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight': 'v',
    }

    def _torch_key_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
        key_maps = dict(_HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map))
        subs = [re.sub(pat, repl, source_key) for pat, repl in key_maps.items() if re.match(pat, source_key)]
        if len(subs) > 1:
            raise ValueError(f'More than 1 key matched: {subs}')
        else:
            return None if len(subs) == 0 else subs[0]

    def _set_param(params, jax_key_str, array):
        jax_key_list = jax_key_str.split('.')
        node = params
        while len(jax_key_list) > 0:
            k = jax_key_list.pop(0)
            if len(jax_key_list) == 0:
                assert array.shape == node[k].shape, f'{jax_key_str}: {array.shape} != {node[k].shape}'
                node[k] = array
            node = node[k]

    # Load tensors directly into the params tree. Q/K/V are the only exception:
    # they need to be fused per layer into Dense_0, so keep just those tensors.
    qkv_by_layer: dict[str, dict[str, np.ndarray]] = {}
    files = list(glob.glob(os.path.join(hf_dir, '*safetensors')))
    for file in files:
        with safe_open(file, framework='numpy') as f:
            for hf_key in f.keys():
                tensor = f.get_tensor(hf_key).astype(np.float32)
                matched_qkv = False
                for pat, role in _QKV_PATTERNS.items():
                    m = re.match(pat, hf_key)
                    if m:
                        layer = m.group(1)
                        qkv_by_layer.setdefault(layer, {})[role] = tensor
                        matched_qkv = True
                        break
                if matched_qkv:
                    continue
                jax_key = _torch_key_to_jax_key(hf_key)
                if jax_key is None:
                    continue
                arr = tensor.T if 'kernel' in jax_key else tensor
                _set_param(params, jax_key, arr)

    # Fuse Q/K/V per layer into Dense_0.
    for layer, qkv in qkv_by_layer.items():
        missing = {'q', 'k', 'v'} - set(qkv)
        if missing:
            raise KeyError(f'Missing QKV tensors for layer {layer}: {sorted(missing)}')
        fused = np.concatenate([qkv['q'], qkv['k'], qkv['v']], axis=0).T
        _set_param(params, f'Block_{layer}.Dense_0.kernel', fused)

    return model, params


def create_model_from_config(
    ckpt_dir: str,
    use_v_head: bool = False,
    use_remat: bool = False,
    use_flash_attn: bool = PLATFORM in ('tpu', 'gpu'),
    mesh: Mesh | None = None,
    tp_size: int = 1,
):
    ckpt_dir = os.path.expanduser(ckpt_dir)
    with open(os.path.join(ckpt_dir, 'config.json')) as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta'],
        mesh=mesh,
        tp_size=tp_size,
        use_v_head=use_v_head,
        use_remat=use_remat,
        use_flash_attn=use_flash_attn,
    )
    _validate_tp_config(model, tp_size)
    return model


def create_param_shape(model: Qwen3Model):
    model = dataclasses.replace(model, mesh=None, tp_size=1, use_flash_attn=False)
    tokens = jnp.ones((1, 1), dtype=jnp.int32)
    token_mask = jnp.ones((1, 1), dtype=jnp.int32)
    return jax.eval_shape(model.init, jax.random.PRNGKey(0), tokens, token_mask)['params']


def orbax_params_dir(ckpt_dir: str, fsdp: bool = False, tp_size: int = 1, precision: str = 'bf16') -> str:
    ckpt_dir = os.path.expanduser(ckpt_dir)
    return os.path.join(ckpt_dir, f'params_orbax_{precision}_fsdp{int(fsdp)}_tp{tp_size}')


def has_orbax_params(ckpt_dir: str, fsdp: bool = False, tp_size: int = 1, precision: str = 'bf16') -> bool:
    params_dir = orbax_params_dir(ckpt_dir, fsdp=fsdp, tp_size=tp_size, precision=precision)
    return os.path.exists(os.path.join(params_dir, '_CHECKPOINT_METADATA'))


def load_orbax_params(
    ckpt_dir: str,
    param_shape,
    param_sharding,
    fsdp: bool = False,
    tp_size: int = 1,
    precision: str = 'bf16',
):
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    dtype = jnp.bfloat16 if precision == 'bf16' else jnp.float32
    params_dir = orbax_params_dir(ckpt_dir, fsdp=fsdp, tp_size=tp_size, precision=precision)

    def make_target(shape, sharding):
        return jax.ShapeDtypeStruct(shape.shape, dtype, sharding=sharding)

    param_shape = flax.linen.unbox(param_shape)
    if isinstance(param_sharding, jax.sharding.Sharding):
        target = jax.tree_util.tree_map(lambda shape: make_target(shape, param_sharding), param_shape)
    else:
        target = jax.tree_util.tree_map(make_target, param_shape, param_sharding)
    print(f'Loading Orbax params from {params_dir}')
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(
        params_dir,
        item=target,
        restore_args=orbax_utils.restore_args_from_target(target),
    )
    print('Loaded Orbax params.')
    return params


_TP_RULES = {
    'Dense_0': 1,  # fused QKV (col-parallel)
    'Dense_1': 0,  # attn out (row-parallel)
    'Dense_2': 1,  # MLP gate (col-parallel)
    'Dense_3': 1,  # MLP up (col-parallel)
    'Dense_4': 0,  # MLP down (row-parallel)
}


def get_tp_spec(path: str, ndim: int) -> tuple:
    if path == 'Dense_0.kernel':
        spec = [None] * ndim
        spec[1] = 'model'
        return tuple(spec)
    if 'Block' not in path or 'kernel' not in path:
        return (None,) * ndim
    for dense_name, tp_axis in _TP_RULES.items():
        if dense_name in path:
            spec = [None] * ndim
            spec[tp_axis] = 'model'
            return tuple(spec)
    return (None,) * ndim


def _validate_tp_config(model: Qwen3Model, tp_size: int):
    if tp_size <= 1:
        return
    assert model.q_heads % tp_size == 0, f'q_heads={model.q_heads} not divisible by tp_size={tp_size}'
    assert model.kv_heads % tp_size == 0, f'kv_heads={model.kv_heads} not divisible by tp_size={tp_size}'
    assert model.mlp_ffw_size % tp_size == 0, f'mlp_ffw_size={model.mlp_ffw_size} not divisible by tp_size={tp_size}'


def _repair_tied_vocab_head(params):
    embed_node = params.setdefault('Embed_0', {})
    head_node = params.setdefault('Dense_0', {})
    embed = embed_node.get('embedding')
    head = head_node.get('kernel')
    embed_missing = embed is None or isinstance(embed, jax.ShapeDtypeStruct)
    head_missing = head is None or isinstance(head, jax.ShapeDtypeStruct)
    if embed_missing and not head_missing:
        embed_node['embedding'] = np.array(head.T, copy=True)
    elif head_missing and not embed_missing:
        head_node['kernel'] = np.array(embed.T, copy=True)
    return params


def create_model_from_ckpt(
    ckpt_dir: str,
    use_v_head: bool = False,
    use_remat: bool = False,
    use_flash_attn: bool = PLATFORM in ('tpu', 'gpu'),
    mesh: Mesh | None = None,
    tp_size: int = 1,
):
    ckpt_dir = os.path.expanduser(ckpt_dir)
    model = create_model_from_config(
        ckpt_dir,
        use_v_head=use_v_head,
        use_remat=use_remat,
        use_flash_attn=use_flash_attn,
        mesh=mesh,
        tp_size=tp_size,
    )
    params_path = os.path.join(ckpt_dir, 'params.pkl')
    print(f'Loading base model params from {params_path}')
    ckpt = Checkpoint(params_path)
    params = ckpt.load_as_dict()['params']
    params = _repair_tied_vocab_head(params)

    # Fuse Q/K/V projections into a single QKV kernel.
    for key in list(params.keys()):
        block = params[key]
        if not isinstance(block, dict):
            continue
        if 'Dense_1' in block and 'Dense_2' in block and 'Dense_6' in block:
            q_kernel = block.pop('Dense_0')['kernel']
            k_kernel = block.pop('Dense_1')['kernel']
            v_kernel = block.pop('Dense_2')['kernel']
            o_proj = block.pop('Dense_3')
            gate_proj = block.pop('Dense_4')
            up_proj = block.pop('Dense_5')
            down_proj = block.pop('Dense_6')
            block['Dense_0'] = {'kernel': np.concatenate([q_kernel, k_kernel, v_kernel], axis=1)}
            block['Dense_1'] = o_proj
            block['Dense_2'] = gate_proj
            block['Dense_3'] = up_proj
            block['Dense_4'] = down_proj

    if use_remat:
        for key in list(params.keys()):
            if 'Block' in key and not key.startswith('Checkpoint'):
                params['Checkpoint' + key] = params.pop(key)

    return model, params
