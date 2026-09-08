import numpy as np

from lmpo.utils.rubric_utils import build_rubric_requests, run_audit

from .rubric_prompts import (
    AUDIT_SYSTEM_PROMPT,
    RUBRIC_SYSTEM_PROMPT,
    STATE_MATCHING_SCAFFOLD_SYSTEM_PROMPT,
    STATE_MATCHING_SYSTEM_PROMPT,
    get_audit_user_prompt,
)


CONFIG = {
    'model_dir': '',
    'inference_batch_per_device': -1,
    'rubric_grade_prompt_length': -1,
    'rubric_num_generation_tokens': 64,
    'state_matching': 0,
    'rubric_scaffold': 0,
    'enable_thinking': 0,
    'n_chunks': 1,
    'do_chunk_by_length': 1,
    'discount': 0.0,
    'fsdp': 0,
    'tp_size': -1,
}


class RubricQwenMetric:
    def rubric_score_qwen(
        self,
        tokens,
        action_mask,
        problems,
        rubrics,
        rubric_n_items,
        inference_batch_per_device,
        tokens_per_action,
        tp_size,
        rubric_grade_prompt_length,
        rubric_num_generation_tokens,
        rng,
        bundle,
        pad_id,
        eos_id,
        state_matching=CONFIG['state_matching'],
        verbose=False,
        rubric_scaffold=CONFIG['rubric_scaffold'],
        enable_thinking=CONFIG['enable_thinking'],
        n_chunks=CONFIG['n_chunks'],
        do_chunk_by_length=CONFIG['do_chunk_by_length'],
        discount=CONFIG['discount'],
        makes_obsoletes=None,
        audit=False,
        **_,
    ):
        import jax

        from .common import sample_qwen_metric_responses

        if verbose:
            print('\n================= RewardMetrics: rubric_score_qwen =================\n')
        rubric_n_items = np.asarray(rubric_n_items)
        assert np.all(rubric_n_items >= 0)
        if makes_obsoletes is not None:
            makes_obsoletes = np.asarray(makes_obsoletes, dtype=object)

        system_prompt = (
            (STATE_MATCHING_SCAFFOLD_SYSTEM_PROMPT if rubric_scaffold else STATE_MATCHING_SYSTEM_PROMPT)
            if state_matching
            else RUBRIC_SYSTEM_PROMPT
        )

        minibatch = inference_batch_per_device * jax.device_count() // tp_size
        auto_mode = rubric_grade_prompt_length <= 0
        B = tokens.shape[0]

        if audit:
            rng, key = jax.random.split(rng)

            def _audit_call(prompts):
                def audit_tokens(prompt):
                    messages = [{'role': 'system', 'content': AUDIT_SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
                    return self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, enable_thinking=False
                    )

                token_lists_audit = [audit_tokens(p) for p in prompts]
                return sample_qwen_metric_responses(
                    token_lists_audit,
                    bundle,
                    self.tokenizer,
                    pad_id,
                    eos_id,
                    num_generation_tokens=512,
                    minibatch=minibatch,
                    rng=key,
                    description='Qwen audit',
                    verbose=verbose,
                    raise_on_truncation=self.raise_on_truncation,
                )[0]

            actions = [self._action_text(tokens, action_mask, i) for i in range(B)]
            audit_prompts = [get_audit_user_prompt(p, r, ro) for p, r, ro in zip(problems, rubrics, actions)]
            run_audit(audit_prompts, _audit_call, source='rubric_score_qwen')

        requests = build_rubric_requests(
            range(B),
            action_mask,
            problems,
            rubrics,
            rubric_n_items,
            makes_obsoletes,
            system_prompt,
            state_matching,
            n_chunks,
            do_chunk_by_length,
            tokens_per_action,
            lambda index, end: self._action_text(
                tokens,
                action_mask,
                index,
                end,
            ),
            self._rubric_user_prompt,
        )
        token_lists = [
            self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f'{system}\n\n{user}'}],
                add_generation_prompt=True,
                enable_thinking=bool(rubric_scaffold) or bool(enable_thinking),
            )
            for _, system, user, _, _ in requests
        ]
        responses, prompt_texts, prompt_truncated = sample_qwen_metric_responses(
            token_lists,
            bundle,
            self.tokenizer,
            pad_id,
            eos_id,
            num_generation_tokens=rubric_num_generation_tokens,
            minibatch=minibatch,
            rng=rng,
            force_length=None if auto_mode else rubric_grade_prompt_length,
            description='rubric_score_qwen',
            verbose=verbose,
            raise_on_truncation=self.raise_on_truncation,
        )
        graded_items = [(request[0], prompt, request[3], request[4]) for request, prompt in zip(requests, prompt_texts)]
        return self._rubric_results(
            graded_items,
            responses,
            [{'prompt_truncated': bool(value)} for value in prompt_truncated],
            n_chunks,
            discount,
        )
