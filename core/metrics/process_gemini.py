import numpy as np

from lmpo.utils.gemini_utils import gemini_parallel, run_gemini_requests
from lmpo.utils.rubric_utils import (
    build_process_reward_items,
    render_metric_outputs,
    run_audit,
    score_process_reward_responses,
)


from .rubric_prompts import (
    PROCESS_REWARDS_AUDIT_SYSTEM_PROMPT,
    PROCESS_REWARDS_SYSTEM_PROMPT,
    get_process_rewards_audit_user_prompt,
    get_process_rewards_user_prompt,
)


CONFIG = {
    'model_name': 'gemini-2.5-flash-lite',
    'thinking_budget': 0,
    'max_output_tokens': 4096,
    'n_chunks': 1,
    'do_chunk_by_length': 1,
    'num_workers': 16,
    'temp': 1.0,
}


class ProcessGeminiMetric:
    def process_rewards_gemini(
        self,
        tokens,
        action_mask,
        problems,
        tokens_per_action,
        model_name=CONFIG['model_name'],
        thinking_budget=CONFIG['thinking_budget'],
        max_output_tokens=CONFIG['max_output_tokens'],
        n_chunks=CONFIG['n_chunks'],
        do_chunk_by_length=CONFIG['do_chunk_by_length'],
        num_workers=CONFIG['num_workers'],
        temp=CONFIG['temp'],
        verbose=False,
        audit=False,
        **_,
    ):
        from lmpo.utils.sharding import host_gather, host_gather_strings_by_process

        from .common import ChunkedValue

        if verbose:
            print('\n================= RewardMetrics: process_rewards_gemini =================\n')

        local_slice, gemini_kw = self._gemini_setup(
            len(tokens), model_name, num_workers, verbose, temp, thinking_budget
        )

        local_problems = [problems[i] for i in local_slice]
        local_items, local_tagged = build_process_reward_items(
            tokens,
            action_mask,
            problems,
            local_slice,
            self.tokenizer,
            get_process_rewards_user_prompt,
        )

        if audit:
            run_audit(
                [get_process_rewards_audit_user_prompt(p, t) for p, t in zip(local_problems, local_tagged)],
                lambda ps: gemini_parallel(
                    ps,
                    PROCESS_REWARDS_AUDIT_SYSTEM_PROMPT,
                    max_output_tokens=512,
                    desc='Gemini audit',
                    **gemini_kw,
                )[0],
                source='process_rewards_gemini',
            )

        local_responses, local_n_attempts, local_failed = run_gemini_requests(
            [item[0] for item in local_items],
            PROCESS_REWARDS_SYSTEM_PROMPT,
            max_output_tokens,
            'Gemini process rewards',
            gemini_kw,
            'process_rewards_gemini',
        )

        local_scores, local_has_formatting = score_process_reward_responses(
            local_responses,
            local_items,
            n_chunks,
            do_chunk_by_length,
            tokens_per_action,
        )

        score_2d = np.asarray(host_gather(np.asarray(local_scores))).reshape(-1, n_chunks)
        scalar_info = {
            'has_formatting': np.asarray(host_gather(np.asarray(local_has_formatting))).reshape(-1),
            'n_attempts': np.asarray(host_gather(np.asarray(local_n_attempts))).reshape(-1),
            'failed': np.asarray(host_gather(np.asarray(local_failed))).reshape(-1),
        }

        infos = {'score': ChunkedValue(score_2d, n_chunks)}
        for chunk_idx in range(1, n_chunks + 1):
            infos[f'chunk_{chunk_idx}_of_{n_chunks}/score'] = score_2d[:, chunk_idx - 1]
        infos.update(scalar_info)

        prompt_texts = [
            f'----- System prompt -----\n{PROCESS_REWARDS_SYSTEM_PROMPT}\n\n----- User prompt -----\n{item[0]}'
            for item in local_items
        ]
        rendered = render_metric_outputs(
            prompt_texts,
            [response or '' for response in local_responses],
            {
                'score': np.asarray(local_scores),
                'has_formatting': np.asarray(local_has_formatting),
                'n_attempts': np.asarray(local_n_attempts),
                'failed': np.asarray(local_failed),
            },
        )
        return (
            infos['score'],
            infos,
            {'rendered': host_gather_strings_by_process(rendered)},
        )
