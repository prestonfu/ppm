import math
import numpy as np
from lmpo.models.tokenizer import token_ids
from lmpo.utils.sharding import host_gather


def auto_force_length(max_len):
    """Compute a padded force_length from the longest sequence in a batch.

    For max_len <= 2/3 * 1024: round 1.5 * max_len up to the nearest power of 2.
    For max_len >  2/3 * 1024: round max_len up to a nearby multiple of 1024.
    """
    if max_len == 0:
        return 0
    threshold = 2 * 1024 // 3  # 682
    if max_len <= threshold:
        return max(128, 2 ** math.ceil(math.log2(1.5 * max_len)))
    else:
        return math.ceil((max_len + 512) / 1024) * 1024


def pad_and_collate(
    token_batch,
    pad_id=0,
    force_length=None,
    how='left',
    description='',
    raise_on_truncation=True,
    frac_too_long_threshold=0.25,
    verbose=False,
):
    """Pad a flat batch with logical shape (batch, variable_length).
    Returns padded tokens shaped (batch, width) and a truncation mask shaped (batch,).
    Fixed widths enforce frac_too_long_threshold.
    """
    assert how in ['left', 'right']
    lens = np.array([len(x) for x in token_batch])
    lens_global = host_gather(lens).flatten() if force_length is None or raise_on_truncation or verbose else lens

    max_input_len = int(lens_global.max())
    max_len = int(force_length) if force_length is not None else auto_force_length(max_input_len)
    frac_too_long = float((lens_global > max_len).mean())
    if force_length is not None and raise_on_truncation and frac_too_long > frac_too_long_threshold:
        raise ValueError(
            f'{description or "token batch"}: {frac_too_long:.1%} of sequences exceed length '
            f'{max_len}; maximum is {max_input_len}'
        )
    if verbose:
        print(
            f'{description or "token batch"}: length min/mean/max='
            f'{lens_global.min()}/{lens_global.mean():.1f}/{max_input_len}, '
            f'p50/p90/p99={np.percentile(lens_global, [50, 90, 99]).astype(int).tolist()}, '
            f'padded_to={max_len}, truncated={frac_too_long:.1%}'
        )

    rows = [
        [pad_id] * (max_len - len(x)) + x if how == 'left' else x + [pad_id] * (max_len - len(x))
        for x in [list(x[:max_len]) for x in token_batch]
    ]
    dtype = token_batch[0].dtype if isinstance(token_batch[0], np.ndarray) else None
    return np.asarray(rows, dtype=dtype), lens > max_len


def round_batch_size(batch_size, k=None):
    batch_size = int(np.ceil(batch_size / 8)) * 8
    if k is not None:
        batch_size = int(np.ceil(batch_size / k)) * k
    return batch_size


def pad_batch_to_round(token_batch, pad_id=0, k=None):
    import jax.numpy as jnp

    module = jnp if isinstance(token_batch, jnp.ndarray) else np
    og_batch_size = token_batch.shape[0]
    batch_size = round_batch_size(og_batch_size, k)
    pad_rows = module.full((batch_size - og_batch_size, *token_batch.shape[1:]), pad_id, dtype=token_batch.dtype)
    padded = module.concatenate([token_batch, pad_rows], axis=0)
    mask = np.arange(batch_size) < og_batch_size
    return padded, mask


def roll_rows(x, shifts):
    B, T = x.shape
    shifts = np.asarray(shifts) % T
    cols = np.arange(T)[None, :]
    src = (cols - shifts[:, None]) % T
    return np.take_along_axis(x, src, axis=1)


def idx_match(tokens, mask, seqs):
    """
    Returns indices of first match subject to mask and which sequence was matched.
    If no match, returns T.
    """
    B, T = tokens.shape
    result = []
    for seq in seqs:
        tok_sliding = np.lib.stride_tricks.sliding_window_view(tokens, len(seq), axis=1)  # (B, T - L + 1, L)
        match = np.all(tok_sliding == seq[None, None, :], axis=-1)  # (B, T - L + 1)
        match = np.concatenate([match, np.zeros((B, len(seq) - 1), dtype=bool)], axis=1)
        match = match & mask
        first_match = np.where(np.any(match, axis=1), np.argmax(match, axis=1), T)
        result.append(first_match)
    result = np.stack(result)
    return np.min(result, axis=0), np.argmin(result, axis=0)


def insert_after(tokens, mask, insert_texts, tokenizer, descriptor, extra_budget=0):
    """
    Inserts insert_texts after the first occurence of search_seq or <eos>, whichever appears first.
    Returns the resulting tokens and token-level insertion mask.

    It is possible to insert tokens outside of mask. Only the start position is
    guaranteed to be within mask.

    If descriptor is 'answer_0.5', then '\n</think>\n\n<answer>' is inserted halfway
    through the thinking trace, and the returned mask applies to 'answer text</answer'.

    Args:
    tokens: (B, T)
    mask: (B, T)
    insert_texts: (B,)
    extra_budget: If > 0, the outputs will have shape (B, T + extra_budget)
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if descriptor.startswith('answer'):
        search_text = ['<', 'answer', '>']
        insert_tok_ls = [tokenizer.encode(t + '</answer') for t in insert_texts]
    elif descriptor.startswith('solution'):
        search_text = ['</think>', '\n\n']
        insert_tok_ls = [tokenizer.encode(t) for t in insert_texts]
    else:
        raise ValueError

    p = 1.0 if '_' not in descriptor else float(descriptor.split('_')[-1])
    pre_insert_text = []
    if descriptor.startswith('answer') and p < 1.0:
        pre_insert_text = ['\n', '</think>', '\n\n', '<', 'answer', '>']
    if descriptor.startswith('solution') and p < 1.0:
        pre_insert_text = ['\n', '</think>', '\n\n']

    B, T = tokens.shape
    ids = token_ids(tokenizer)
    search_toks = np.array([ids[x] for x in search_text])
    pre_insert_toks = [ids[x] for x in pre_insert_text]
    insert_toks, _ = pad_and_collate(
        [pre_insert_toks + x for x in insert_tok_ls],
        how='right',
        pad_id=pad_id,
        force_length=T + extra_budget,
    )
    insert_lengths = np.array([len(x) for x in insert_tok_ls])

    match_idxs, match_seq_idxs = idx_match(tokens, mask, [search_toks, np.array([eos_id])])
    should_insert = match_idxs != T
    match_idxs = np.where(should_insert, (match_idxs * p).astype(int), T)
    start = np.where(should_insert, np.where(match_seq_idxs == 0, match_idxs + len(search_toks), match_idxs), T)
    eff_start = np.where(should_insert, start + len(pre_insert_toks), start)
    eff_insert_lengths = np.minimum(insert_lengths, T + extra_budget - start)

    tokens = np.concatenate([tokens, np.full((B, extra_budget), pad_id, dtype=int)], axis=1)
    col_idx = np.arange(T + extra_budget)[None, :]
    out = roll_rows(tokens, -start)
    out = np.where(should_insert[:, None] & (col_idx < eff_insert_lengths[:, None]), insert_toks, out)
    out = roll_rows(out, start)
    inserted_mask = (
        should_insert[:, None] & (col_idx >= eff_start[:, None]) & (col_idx < (start + eff_insert_lengths)[:, None])
    )
    return out, inserted_mask


def remove_left_padding(tokens, pad_id):
    tokens = np.asarray(tokens)
    assert tokens.ndim == 1
    return tokens[np.argmax(tokens != pad_id) :]
