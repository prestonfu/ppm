import dataclasses

import jax
import numpy as np
from tqdm import tqdm

from lmpo.core.sampling import Sampler
from lmpo.envs.base import clean_action
from lmpo.utils.array_utils import pad_and_collate, pad_batch_to_round, remove_left_padding
from lmpo.utils.rubric_utils import score_rubric_outputs
from lmpo.utils.sharding import host_gather

from .rubric_prompts import get_rubric_grade_user_prompt, get_state_matching_user_prompt


@dataclasses.dataclass(frozen=True)
class ChunkedValue:
    values: np.ndarray
    n_chunks: int

    def __post_init__(self):
        assert self.values.shape[-1] == self.n_chunks


def combine_turn_results(turn_results):
    combined = {}
    for name in turn_results[0]:
        values = [result[name] for result in turn_results]
        first = values[0]
        if isinstance(first, ChunkedValue):
            combined[name] = ChunkedValue(
                np.stack([np.asarray(v.values, dtype=np.float32) for v in values], axis=1),
                first.n_chunks,
            )
        else:
            combined[name] = np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=1)[..., None]
    return combined


@dataclasses.dataclass
class ModelBundle:
    model: object
    shard_params: object
    data_shard: object
    no_shard: object
    local_slice: object
    shard_data: object
    sampler: object = None
    cpu_params: object = None
    params: object = None
    uses_policy_params: bool = False

    def load(self, tokenizer, base_params, sampling_params, sampler):
        if self.uses_policy_params:
            self.params = sampling_params if sampling_params is not None else self.shard_params(base_params)
            self.sampler = sampler or self.sampler
        elif self.params is None:
            self.params = self.shard_params(self.cpu_params)

        if self.sampler is None:
            self.sampler = Sampler(self.model, tokenizer, self.data_shard, self.no_shard)

    def release(self):
        if not self.uses_policy_params and self.params is not None:
            for value in jax.tree.leaves(self.params):
                delete = getattr(value, 'delete', None)
                if delete is not None:
                    delete()
        self.params = None


def sample_qwen_metric_responses(
    token_lists,
    bundle,
    tokenizer,
    pad_id,
    eos_id,
    num_generation_tokens,
    minibatch,
    rng,
    description,
    verbose,
    raise_on_truncation,
    force_length=None,
):
    prompt_tokens, prompt_truncated = pad_and_collate(
        token_lists,
        pad_id=pad_id,
        force_length=force_length,
        description=f'{description} prompt',
        raise_on_truncation=raise_on_truncation,
        verbose=verbose,
    )
    prompt_batches, keep = pad_batch_to_round(prompt_tokens, pad_id, minibatch)
    prompt_batches = prompt_batches.reshape(-1, minibatch, prompt_batches.shape[-1])
    keep = keep.reshape(-1, minibatch)
    prompt_truncated, _ = pad_batch_to_round(prompt_truncated, False, minibatch)
    prompt_truncated = prompt_truncated.reshape(-1, minibatch)
    responses, prompts, prompt_truncated_list = [], [], []

    for batch_index in tqdm(range(len(prompt_batches)), desc=description, disable=not verbose):
        prompt_local = bundle.shard_data(bundle.local_slice(prompt_batches[batch_index]))
        rng, key = jax.random.split(rng)
        state = bundle.sampler.sample(
            bundle.params,
            prompt_local,
            num_generation_tokens,
            key,
            max_seq_len=prompt_local.shape[1] + num_generation_tokens,
            verbose=verbose,
            timer_name=f'reward/{description}',
        )
        tokens = np.asarray(host_gather(state.tokens))
        action_idx = np.asarray(host_gather(state.action_idx))
        for index, output_tokens in enumerate(tokens):
            if not keep[batch_index, index]:
                continue
            action, _ = clean_action(output_tokens[action_idx[index] == 0].tolist(), eos_id)
            prompt = remove_left_padding(prompt_batches[batch_index, index], pad_id)
            responses.append(tokenizer.decode(action))
            prompts.append(tokenizer.decode(prompt.tolist()))
            prompt_truncated_list.append(bool(prompt_truncated[batch_index, index]))

    return responses, prompts, prompt_truncated_list


class MetricHelper:
    @staticmethod
    def _aggregate_metrics_str(infos):
        """Build an aggregate-metrics string (mean over batch) for scalar-valued infos."""
        lines = []
        for k, v in sorted(infos.items()):
            arr = np.array(v)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                lines.append(f'{k} (mean): {arr.mean():.4g}')
        return '\n'.join(lines)

    def _action_text(self, tokens, action_mask, i, n_keep=None):
        action_toks = tokens[i][action_mask[i].astype(bool)]
        if n_keep is not None:
            action_toks = action_toks[:n_keep]
        return self.tokenizer.decode(action_toks.tolist())

    @staticmethod
    def _gemini_setup(n_total, model, num_workers, verbose, temp, thinking_budget):
        pc = jax.process_count()
        assert n_total % pc == 0, f'batch size {n_total} not divisible by process_count {pc}'
        per_host = n_total // pc
        local_slice = np.arange(jax.process_index() * per_host, (jax.process_index() + 1) * per_host)
        return local_slice, dict(
            model=model, num_workers=num_workers, verbose=verbose, temp=temp, thinking_budget=thinking_budget
        )

    @staticmethod
    def _rubric_user_prompt(problem, rubric, action_text, state_matching):
        def _strip(text):
            for tok in ['<|im_start|>', '<|im_end|>', '<think>', '</think>']:
                text = text.replace(tok, '')
            return text

        if state_matching:
            return get_state_matching_user_prompt(problem, rubric, _strip(action_text))
        solution_text = (
            action_text.split('</think>')[1].strip()
            if '</think>' in action_text
            else 'This student did not provide a solution.'
        )
        return get_rubric_grade_user_prompt(problem, rubric, _strip(solution_text))

    def _rubric_results(self, items, responses, diagnostics, n_chunks, discount, system_prompt=None, gather=False):
        infos, renders = score_rubric_outputs(items, responses, diagnostics, n_chunks, discount, system_prompt, gather)
        infos['reward'] = ChunkedValue(infos['reward'], n_chunks)
        infos['value'] = ChunkedValue(infos['value'], n_chunks)
        return infos['value'], infos, renders
