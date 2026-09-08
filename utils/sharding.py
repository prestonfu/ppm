import functools

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils as mu
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh
from lmpo.utils.jax_utils import ns


@functools.lru_cache(maxsize=32)
def get_mesh(device_count: int, tp_size: int) -> Mesh:
    """2-D mesh: batch axis ``data``, optional tensor-parallel axis ``model`` (size ``tp_size``)."""
    assert device_count % tp_size == 0, f'device_count={device_count} not divisible by tp_size={tp_size}'
    data_size = device_count // tp_size
    device_mesh = mesh_utils.create_device_mesh((data_size, tp_size), allow_split_physical_axes=True)
    return Mesh(devices=device_mesh, axis_names=('data', 'model'))


def create_sharding(train_state_shape, fsdp: bool = False, tp_size: int = 1, get_tp_spec=None):
    if tp_size == 1:
        get_tp_spec = None
    assert jax.device_count() % tp_size == 0, f'device_count={jax.device_count()} not divisible by tp_size={tp_size}'
    data_size = jax.device_count() // tp_size
    mesh = get_mesh(jax.device_count(), tp_size)
    data_sharding = ns(mesh, 'data')
    no_shard = ns(mesh)

    def path_str(path):
        parts = []
        for p in path:
            if hasattr(p, 'key'):
                parts.append(str(p.key))
            elif hasattr(p, 'idx'):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        return '.'.join(parts)

    min_size_bytes = 4 * (2**20)

    def shard_parameter(param, path_str):
        all_nones = (None,) * param.ndim
        if np.prod(param.shape) * param.dtype.itemsize <= min_size_bytes:
            return all_nones
        tp_spec = get_tp_spec(path_str, param.ndim) if get_tp_spec is not None else all_nones
        if not fsdp:
            return tp_spec if any(s is not None for s in tp_spec) else all_nones
        for i in range(param.ndim):
            if tp_spec[i] is None and param.shape[i] % data_size == 0:
                return tuple('data' if j == i else tp_spec[j] for j in range(param.ndim))
        if any(s is not None for s in tp_spec):
            return tp_spec
        print(f'Could not shard parameter of shape {param.shape}. Defaulting to full replication.')
        return all_nones

    if not fsdp and get_tp_spec is None:
        train_state_sharding = no_shard
    else:
        train_state_sharding = jax.tree_util.tree_map_with_path(
            lambda path, spec: ns(mesh, *shard_parameter(spec, path_str(path))),
            flax.linen.unbox(train_state_shape),
        )

    device_id_to_data_idx = {mesh.devices[di, ti].id: di for di in range(data_size) for ti in range(tp_size)}
    local_data_indices = sorted({device_id_to_data_idx[d.id] for d in mesh.local_devices})

    def get_local_slice(x):
        """Take the global batch on the ``data`` mesh axis and return this process's local slice."""
        assert isinstance(x, (np.ndarray, jnp.ndarray)), f'Got type {type(x)}'
        if x.shape[0] % data_size != 0:
            assert False, f'{x.shape[0]} % {data_size} != 0'
        shard_size = x.shape[0] // data_size
        slices = [x[i * shard_size : (i + 1) * shard_size] for i in local_data_indices]
        return np.concatenate(slices, axis=0) if isinstance(x, np.ndarray) else jnp.concatenate(slices, axis=0)

    def shard_data(*args):
        def _shard_data(x):
            if jax.local_device_count() == jax.device_count():
                return jax.device_put(x, data_sharding)
            assert x.shape[0] % len(local_data_indices) == 0, f'{x.shape[0]} % {len(local_data_indices)} != 0'
            shard_size = x.shape[0] // len(local_data_indices)
            global_shape = (shard_size * data_size, *x.shape[1:])
            return jax.make_array_from_process_local_data(data_sharding, x, global_shape)

        if len(args) == 1:
            return _shard_data(args[0])
        return jax.tree_util.tree_map(_shard_data, args)

    return train_state_sharding, no_shard, data_sharding, shard_data, get_local_slice


def get_shard_params_fn(params_shard, no_shard, precision=jnp.bfloat16):
    to_dtype = lambda t: jax.tree_util.tree_map(lambda x: x.astype(precision), t)
    if params_shard == no_shard:

        def shard_params_fn(params):
            return jax.device_put(to_dtype(params), no_shard)

        return shard_params_fn

    @functools.partial(jax.jit, out_shardings=params_shard)
    def jitted(p):
        print('JIT compiling shard_params (to_dtype)')
        return to_dtype(p)

    def shard_params_fn(params):
        for get_p in [lambda: params, lambda: host_gather_leaves(params)]:
            try:
                result = jitted(get_p())
                return result
            except ValueError as e:
                if 'incompatible devices' not in str(e):
                    raise
                last = e
        raise last

    return shard_params_fn


def host_gather(x):
    if jax.process_count() > 1:
        return process_allgather(x)
    return x


def host_gather_bytes(payloads: list[bytes], shard_data_fn) -> list[bytes]:
    """Gather variable-length bytes in mesh data-axis row order."""
    if not payloads:
        raise ValueError('cannot gather an empty byte batch')

    rows = [np.frombuffer(bytes(payload), dtype=np.uint8) for payload in payloads]
    lengths_local = np.asarray([len(row) for row in rows], dtype=np.int32)
    lengths_by_process = np.asarray(host_gather(lengths_local)).reshape(-1)
    max_length = int(lengths_by_process.max())

    padded_local = np.zeros((len(rows), max_length), dtype=np.uint8)
    for i, row in enumerate(rows):
        padded_local[i, : len(row)] = row

    bytes_global = np.asarray(host_gather(shard_data_fn(padded_local)))
    lengths_global = np.asarray(host_gather(shard_data_fn(lengths_local))).reshape(-1)
    return [bytes(row[: int(length)]) for row, length in zip(bytes_global, lengths_global)]


def host_local_slice(x):
    """This process's rows of a batch-sharded jax.Array, as numpy. Like get_local_slice(host_gather(x))
    but reads locally addressable shards directly, with no interhost communication."""
    if isinstance(x, np.ndarray):
        return x
    if jax.process_count() == 1:
        return np.asarray(x)
    shards = {}
    for s in x.addressable_shards:
        start = s.index[0].start or 0
        if start not in shards:
            shards[start] = np.asarray(s.data)
    parts = [shards[k] for k in sorted(shards)]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def host_gather_sum(scalar):
    return int(np.asarray(host_gather(np.array([scalar]))).sum())


def host_gather_leaves(x):
    """Like host_gather but gathers one leaf at a time to avoid peak-memory spikes."""
    if jax.process_count() == 1:
        return x
    leaves, treedef = jax.tree_util.tree_flatten(x)
    gathered = [process_allgather(leaf) for leaf in leaves]
    return treedef.unflatten(gathered)


def host_gather_strings_by_process(strings: list[str]) -> list[str]:
    """Gather strings across hosts using process-index-based partitioning (no mesh sharding).
    Requires every host to call with the same number of strings."""
    if jax.process_count() == 1:
        return strings
    from lmpo.utils.array_utils import pad_and_collate

    encoded = [np.frombuffer(s.encode('utf-8'), dtype=np.uint8) for s in strings]
    padded, _ = pad_and_collate(encoded, how='right')
    lens = np.array([len(e) for e in encoded])
    gathered_bytes = np.asarray(host_gather(padded)).reshape(-1, padded.shape[-1])
    gathered_lens = np.asarray(host_gather(lens)).reshape(-1)
    return [bytes(gathered_bytes[i, : gathered_lens[i]]).decode('utf-8') for i in range(len(gathered_lens))]


def broadcast_str_process0(value: str | None) -> str:
    if jax.process_index() == 0:
        if value is None:
            raise ValueError('process 0 must provide a string')
        data = np.frombuffer(value.encode('utf-8'), dtype=np.uint8)
        n = np.array([data.size], dtype=np.int32)
    else:
        data = np.empty(0, dtype=np.uint8)
        n = np.array([0], dtype=np.int32)

    n = int(np.asarray(mu.broadcast_one_to_all(n))[0])
    if jax.process_index() != 0:
        data = np.zeros(n, dtype=np.uint8)

    data = np.asarray(mu.broadcast_one_to_all(data))
    return bytes(data).decode('utf-8')


def get_memory_usage():
    return get_memory_stats()['in_use_gib']


def get_memory_stats():
    stats = jax.local_devices()[0].memory_stats() or {}  # None on CPU backends
    bytes_in_use = stats.get('bytes_in_use', 0)
    bytes_limit = stats.get('bytes_limit')
    peak_bytes_in_use = stats.get('peak_bytes_in_use', bytes_in_use)
    available_bytes = None if bytes_limit is None else max(bytes_limit - bytes_in_use, 0)
    gib = 1024**3
    return {
        'bytes_in_use': bytes_in_use,
        'bytes_limit': bytes_limit,
        'peak_bytes_in_use': peak_bytes_in_use,
        'available_bytes': available_bytes,
        'in_use_gib': bytes_in_use / gib,
        'limit_gib': None if bytes_limit is None else bytes_limit / gib,
        'peak_gib': peak_bytes_in_use / gib,
        'available_gib': None if available_bytes is None else available_bytes / gib,
    }


def format_memory_summary(stats, sampling_base_gib=None):
    summary = [
        f'in_use={stats["in_use_gib"]:.2f} GiB',
        f'peak={stats["peak_gib"]:.2f} GiB',
    ]
    if stats['limit_gib'] is not None:
        summary.append(f'limit={stats["limit_gib"]:.2f} GiB')
        summary.append(f'free={stats["available_gib"]:.2f} GiB')
    if sampling_base_gib is not None:
        sampling_used_gib = max(stats['in_use_gib'] - sampling_base_gib, 0.0)
        summary.append(f'sampling_used={sampling_used_gib:.2f} GiB')
        if stats['limit_gib'] is not None:
            sampling_budget_gib = max(stats['limit_gib'] - sampling_base_gib, 0.0)
            summary.append(f'sampling_budget={sampling_budget_gib:.2f} GiB')
            summary.append(f'sampling_free={max(stats["limit_gib"] - stats["in_use_gib"], 0.0):.2f} GiB')
    return ', '.join(summary)
