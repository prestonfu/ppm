import numpy as np

from lmpo.utils.gemini_utils import gemini_parallel, run_gemini_requests
from lmpo.utils.rubric_utils import build_rubric_requests, run_audit

from .rubric_prompts import (
    AUDIT_SYSTEM_PROMPT,
    RUBRIC_SYSTEM_PROMPT,
    STATE_MATCHING_SCAFFOLD_SYSTEM_PROMPT,
    STATE_MATCHING_SYSTEM_PROMPT,
    get_audit_user_prompt,
)


CONFIG = {
    'model_name': 'gemini-2.5-flash-lite',
    'state_matching': 1,
    'rubric_scaffold': 1,
    'thinking_budget': 0,
    'max_output_tokens': 2048,
    'n_chunks': 1,
    'do_chunk_by_length': 0,
    'discount': 0.0,
    'num_workers': 16,
    'temp': 1.0,
}


class RubricGeminiMetric:
    def rubric_score_gemini(
        self,
        tokens,
        action_mask,
        problems,
        rubrics,
        rubric_n_items,
        tokens_per_action,
        model_name=CONFIG['model_name'],
        state_matching=CONFIG['state_matching'],
        rubric_scaffold=CONFIG['rubric_scaffold'],
        thinking_budget=CONFIG['thinking_budget'],
        max_output_tokens=CONFIG['max_output_tokens'],
        n_chunks=CONFIG['n_chunks'],
        do_chunk_by_length=CONFIG['do_chunk_by_length'],
        discount=CONFIG['discount'],
        num_workers=CONFIG['num_workers'],
        temp=CONFIG['temp'],
        makes_obsoletes=None,
        verbose=False,
        audit=False,
        **_,
    ):
        if verbose:
            print('\n================= RewardMetrics: rubric_score_gemini =================\n')
        rubric_n_items = np.asarray(rubric_n_items)
        assert np.all(rubric_n_items >= 0)
        if makes_obsoletes is not None:
            makes_obsoletes = np.asarray(makes_obsoletes, dtype=object)

        system_prompt = (
            (STATE_MATCHING_SCAFFOLD_SYSTEM_PROMPT if rubric_scaffold else STATE_MATCHING_SYSTEM_PROMPT)
            if state_matching
            else RUBRIC_SYSTEM_PROMPT
        )

        local_slice, gemini_kw = self._gemini_setup(
            len(tokens), model_name, num_workers, verbose, temp, thinking_budget
        )

        if audit:
            audit_problems = [problems[i] for i in local_slice]
            audit_rollouts = [self._action_text(tokens, action_mask, int(i)) for i in local_slice]
            audit_rubrics = [rubrics[i] for i in local_slice]
            run_audit(
                [get_audit_user_prompt(p, r, ro) for p, r, ro in zip(audit_problems, audit_rubrics, audit_rollouts)],
                lambda ps: gemini_parallel(
                    ps,
                    AUDIT_SYSTEM_PROMPT,
                    max_output_tokens=512,
                    desc='Gemini audit',
                    **gemini_kw,
                )[0],
                source='rubric_score_gemini',
            )

        local_items = [
            (chunk, user, n_items, obsolete)
            for chunk, _, user, n_items, obsolete in build_rubric_requests(
                local_slice,
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
        ]

        # Dedup identical prompts (e.g. trailing chunks of a short response that all
        # clamp to the same cutoff) so the judge is queried once per distinct prefix.
        uniq_prompts, inverse = {}, []
        for it in local_items:
            inverse.append(uniq_prompts.setdefault(it[1], len(uniq_prompts)))
        uniq_responses, uniq_n_attempts, uniq_failed = run_gemini_requests(
            list(uniq_prompts),
            system_prompt,
            max_output_tokens,
            'Gemini rubric grading',
            gemini_kw,
            'rubric_score_gemini',
        )
        local_responses = [uniq_responses[j] for j in inverse]
        local_n_attempts = [uniq_n_attempts[j] for j in inverse]
        local_failed = [uniq_failed[j] for j in inverse]

        diagnostics = [
            {'n_attempts': attempts, 'failed': failed} for attempts, failed in zip(local_n_attempts, local_failed)
        ]
        return self._rubric_results(
            local_items,
            [response or '' for response in local_responses],
            diagnostics,
            n_chunks,
            discount,
            system_prompt,
            gather=True,
        )
