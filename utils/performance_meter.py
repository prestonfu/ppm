import json
import os


class PerformanceMeter:
    _sum = {}
    _count = {}
    _snapshot_sum = {}
    _snapshot_count = {}
    _mode = {}

    @classmethod
    def add(cls, name, value):
        cls._mode.setdefault(name, 'sum')
        assert cls._mode[name] == 'sum', f'{name} was previously registered as {cls._mode[name]}'
        cls._sum[name] = cls._sum.get(name, 0) + value
        cls._count[name] = cls._count.get(name, 0) + 1

    @classmethod
    def average(cls, name, value):
        cls._mode.setdefault(name, 'average')
        assert cls._mode[name] == 'average', f'{name} was previously registered as {cls._mode[name]}'
        cls._sum[name] = cls._sum.get(name, 0) + value
        cls._count[name] = cls._count.get(name, 0) + 1

    @classmethod
    def get(cls):
        """Window values since the last snapshot: sum for 'sum' metrics, mean for 'average'."""
        out = {}
        for name, mode in cls._mode.items():
            total = cls._sum.get(name, 0) - cls._snapshot_sum.get(name, 0)
            count = cls._count.get(name, 0) - cls._snapshot_count.get(name, 0)
            if count == 0:
                continue
            out[name] = total if mode == 'sum' else total / count
        return out

    @classmethod
    def total(cls, prefix):
        return sum(value for name, value in cls._sum.items() if name.startswith(prefix))

    @classmethod
    def get_log_metrics(cls):
        """Return window and cumulative performance metrics, then start a new window."""
        metrics = {f'perf/{name}': value for name, value in cls.get().items()}
        tokens_in = cls.total('tokens_in/')
        tokens_out = cls.total('tokens_out/')
        metrics.update(
            {
                'perf/flops_cumulative': cls.total('flops_'),
                'perf/tokens_in/cumulative': tokens_in,
                'perf/tokens_out/cumulative': tokens_out,
                'perf/tokens_thinking/cumulative': cls.total('tokens_thinking/'),
                'perf/tokens_total/cumulative': cls.total('tokens_total/'),
                'perf/tokens/cumulative': tokens_in + tokens_out,
                'perf/gemini_cost/cumulative': cls.total('gemini_cost/'),
            }
        )
        cls.take_snapshot()
        return metrics

    @classmethod
    def take_snapshot(cls):
        cls._snapshot_sum = dict(cls._sum)
        cls._snapshot_count = dict(cls._count)

    @classmethod
    def export_state(cls):
        return {
            'sum': dict(cls._sum),
            'count': dict(cls._count),
            'mode': dict(cls._mode),
            'snapshot_sum': dict(cls._snapshot_sum),
            'snapshot_count': dict(cls._snapshot_count),
        }

    @classmethod
    def import_state(cls, state):
        if not state:
            return
        cls._sum = dict(state['sum'])
        cls._count = dict(state['count'])
        cls._mode = dict(state['mode'])
        cls._snapshot_sum = dict(state.get('snapshot_sum', {}))
        cls._snapshot_count = dict(state.get('snapshot_count', {}))


def model_flops_per_token(ckpt_dir):
    """
    Compute exact matmul FLOPs for one forward pass token through a Qwen3 model.

    Returns (flops_per_token, flops_prefill_fn) where:
      - flops_per_token: FLOPs for a single decode step (matmuls only, no attn scores)
      - flops_prefill_fn(B, T): FLOPs for a prefill over B sequences of length T
        (includes attention score matmuls QK^T and AV)

    For training (forward + backward ≈ 3 * forward), multiply flops_per_token by 3.
    """
    with open(os.path.join(os.path.expanduser(ckpt_dir), 'config.json')) as f:
        cfg = json.load(f)

    hidden = cfg['hidden_size']
    q_heads = cfg['num_attention_heads']
    kv_heads = cfg['num_key_value_heads']
    head_dim = cfg['head_dim']
    ffw = cfg['intermediate_size']
    vocab = cfg['vocab_size']
    num_layers = cfg['num_hidden_layers']

    # Per-layer matmul FLOPs per token (no attention score ops)
    attn_proj = 2 * hidden * (q_heads + 2 * kv_heads) * head_dim  # Q + K + V projections
    attn_proj += 2 * q_heads * head_dim * hidden  # output projection
    mlp = 2 * hidden * ffw * 3  # gate + up + down
    per_layer = attn_proj + mlp

    lm_head = 2 * hidden * vocab

    flops_per_token = num_layers * per_layer + lm_head

    def flops_prefill(B, T):
        """FLOPs for prefilling B sequences of length T (includes QK^T + AV attention)."""
        attn_scores_per_layer = 4 * T * q_heads * head_dim  # per token: QK^T and AV
        return B * T * (flops_per_token + num_layers * attn_scores_per_layer)

    return flops_per_token, flops_prefill
