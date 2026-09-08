import os

from jax.sharding import Mesh, NamedSharding, PartitionSpec

P = PartitionSpec


def ns(mesh: Mesh, *components) -> NamedSharding:
    if not components:
        return NamedSharding(mesh, P())
    return NamedSharding(mesh, P(*components))


def init_jax_compilation_cache(cache_dir: str | None = None) -> None:
    """Point JAX at an on-disk compilation cache. No-op if the API is missing."""
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        path = os.path.expanduser(cache_dir or '~/jax-cache')
        os.makedirs(path, exist_ok=True)
        cc.set_cache_dir(path)
    except Exception:
        pass
