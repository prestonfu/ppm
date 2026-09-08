import numpy as np
from tqdm import tqdm

from lmpo.utils.array_utils import insert_after, pad_batch_to_round


CONFIG = {
    'model_dir': '',
    'inference_batch_per_device': -1,
    'fsdp': 0,
    'tp_size': -1,
    'extra_budget': 4096,
}


class VerifreeMetric:
    def verifree(
        self,
        tokens,
        prompt_mask,
        action_mask,
        oracle_solutions,
        prompt_length,
        tokens_per_action,
        inference_batch_per_device,
        tp_size,
        bundle,
        num_turns,
        env_enable_thinking,
        pad_id,
        extra_budget=CONFIG['extra_budget'],
        verbose=False,
        **_,
    ):
        import jax
        import jax.numpy as jnp

        from lmpo.utils.sharding import format_memory_summary, get_memory_stats

        assert num_turns == 1, f'verifree currently only supports bandit envs (num_turns=1), got num_turns={num_turns}'
        params = bundle.params
        minibatch = inference_batch_per_device * jax.device_count() // tp_size

        B, T = tokens.shape
        prompt_lens = prompt_mask.sum(axis=1).astype(np.int32)
        action_lens = action_mask.sum(axis=1).astype(np.int32)

        # Prompt: left-pad to prompt_length. src_pos = col - (prompt_length - prompt_lens); negative => pad.
        col_p = np.arange(prompt_length, dtype=np.int32)[None, :]
        src_pos = col_p - (prompt_length - prompt_lens[:, None])
        prompt_tokens = np.take_along_axis(tokens, np.clip(src_pos, 0, T - 1), axis=1)
        prompt_tokens = np.where(src_pos >= 0, prompt_tokens, pad_id).astype(tokens.dtype)

        if env_enable_thinking == 0:
            # No thinking: oracle solution sits directly after the prompt; no insert_after needed.
            width = tokens_per_action + extra_budget
            sol_tok_lists = [self.tokenizer.encode(s) for s in oracle_solutions]
            sol_lengths = np.array([min(len(x), width) for x in sol_tok_lists], dtype=np.int32)
            tokens_with_answers = np.full((B, width), pad_id, dtype=tokens.dtype)
            for i, toks in enumerate(sol_tok_lists):
                k = sol_lengths[i]
                tokens_with_answers[i, :k] = toks[:k]
            answer_mask = np.arange(width)[None, :] < sol_lengths[:, None]
        else:
            # Action: starts at prompt_lens, right-pad to tokens_per_action.
            col_a = np.arange(tokens_per_action, dtype=np.int32)[None, :]
            action_idx = np.clip(prompt_lens[:, None] + col_a, 0, T - 1)
            action_tokens = np.take_along_axis(tokens, action_idx, axis=1)
            action_mask_compact = col_a < action_lens[:, None]
            action_tokens = np.where(action_mask_compact, action_tokens, pad_id).astype(tokens.dtype)

            tokens_with_answers, answer_mask = insert_after(
                action_tokens,
                action_mask_compact,
                oracle_solutions,
                self.tokenizer,
                descriptor='solution',
                extra_budget=extra_budget,
            )

        tokens_with_answers = np.concatenate([prompt_tokens, tokens_with_answers], axis=1)
        answer_mask = np.concatenate([np.zeros_like(prompt_tokens, dtype=bool), answer_mask], axis=1)
        answer_length = answer_mask.sum(axis=1)

        def reshaper(x):
            x, _ = pad_batch_to_round(x, pad_id if x.dtype == np.int32 else 0, minibatch)
            return x.reshape(-1, minibatch, *x.shape[1:])

        _, batch_mask = pad_batch_to_round(tokens_with_answers, pad_id, minibatch)
        batch_mask = batch_mask.reshape(-1, minibatch)
        all_tokens_batched = reshaper(tokens_with_answers)
        all_masks_batched = reshaper(answer_mask)

        all_logprobs_list = []
        for mini_i in tqdm(range(len(all_tokens_batched)), desc='reward_metrics/verifree', disable=not verbose):
            if verbose and mini_i < 2:
                print(f'[verifree mini_i={mini_i}] {format_memory_summary(get_memory_stats())}')
            tokens_local = bundle.local_slice(all_tokens_batched[mini_i])
            mask_local = bundle.local_slice(all_masks_batched[mini_i])
            logprobs = bundle.sampler.logprobs_for_tokens(
                params,
                bundle.shard_data(tokens_local),
                bundle.shard_data(mask_local.astype(jnp.int32)),
            )
            all_logprobs_list.append(logprobs[batch_mask[mini_i]])

        all_logprobs = np.maximum(np.concatenate(all_logprobs_list, axis=0), -10)
        answer_logprob = all_logprobs.sum(axis=1)
        logprob_norm = np.where(answer_length > 0, answer_logprob / (answer_length + 1e-8), -10)

        infos = {
            'logprob': np.where(answer_length > 0, answer_logprob, -10),
            'logprob_norm': logprob_norm,
            'prob': np.where(answer_length > 0, np.exp(answer_logprob), 0),
            'prob_norm': np.exp(logprob_norm),
            'length': answer_length,
        }
        agg_str = self._aggregate_metrics_str(infos)
        rendered = []
        for i in range(len(tokens_with_answers)):
            row = tokens_with_answers[i]
            mask_row = answer_mask[i]
            if mask_row.any():
                splice_start = int(np.argmax(mask_row))
                context_toks = [int(t) for t in row[:splice_start] if int(t) != pad_id]
                scored = self.tokenizer.decode(row[mask_row].tolist())
            else:
                context_toks = [int(t) for t in row if int(t) != pad_id]
                scored = ''
            context = self.tokenizer.decode(context_toks)
            metrics_str = '\n'.join(f'{k}: {np.array(v)[i]:.4g}' for k, v in sorted(infos.items()))
            rendered.append(
                f'Given this context:\n\n{context}\n\n'
                f'We are measuring the likelihood of the following tokens:\n\n{scored}\n\n'
                f'----- Metrics -----\n{metrics_str}\n\n'
                f'----- Aggregate metrics -----\n{agg_str}'
            )
        return (
            infos['prob_norm'],
            infos,
            {'rendered': np.array(rendered, dtype=object)},
        )
