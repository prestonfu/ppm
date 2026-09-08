import os

import jax
import jax.numpy as jnp
import numpy as np

from lmpo.core.sampling import TokenRole
from lmpo.core.metrics.rubric_prompts import OPSD_AUDIT_SYSTEM_PROMPT, get_opsd_audit_user_prompt
from lmpo.utils.array_utils import pad_and_collate
from lmpo.utils.gemini_utils import gemini_parallel
from lmpo.utils.rubric_utils import run_audit


def build_opsd_prompts(tokenizer, env, problems, opsd_contexts, opsd_prompt_length):
    """Build teacher-conditioned OPSD prompts and a per-example validity mask."""
    B = len(problems)
    aug_lists = [None] * B
    opsd_mask = np.zeros(B, dtype=bool)
    enable_thinking = bool(int(getattr(env, 'enable_thinking', 1)))

    for i in range(B):
        problem = str(problems[i]).strip()
        assert problem, f'problem is empty for example {i} - env must provide problem when OPSD is enabled'
        ctx = (str(opsd_contexts[i]) if opsd_contexts[i] is not None else '').strip()
        assert ctx, f'opsd_context is empty for example {i} - env must provide opsd_context when OPSD is enabled'

        suffix = (
            '\n\nAfter understanding the reference solution, '
            'please try to solve this problem using your own approach below.'
        )
        build = lambda c: tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f'{problem}\n\nHere is a reference solution:\n\n{c}{suffix}'}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        toks = None
        while True:
            candidate = build(ctx)
            if len(candidate) <= opsd_prompt_length:
                toks = candidate
                break
            if not ctx:
                break
            nl = ctx.rfind('\n')
            ctx = ctx[:nl] if nl != -1 else ''

        if toks is not None:
            opsd_mask[i] = True
        else:
            fallback = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': problem}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            toks = fallback[-opsd_prompt_length:]
        aug_lists[i] = list(toks)

    return aug_lists, opsd_mask


def build_opsd_training_batch(
    tokenizer,
    env,
    states,
    all_tokens,
    all_roles,
    all_action_idx,
    opsd_prompt_length,
    num_generation_tokens,
    pad_id,
    allow_prompt_truncation,
    verbose=False,
    audit=False,
    audit_model_name='gemini-2.5-flash-lite',
    audit_num_workers=16,
):
    """Pack base-teacher OPSD inputs and map student action columns to teacher sequence columns."""
    opsd_seq_len = int(opsd_prompt_length) + int(num_generation_tokens)
    problems = [s.problem for s in states]
    opsd_contexts = [s.opsd_context for s in states]
    aug_lists, opsd_mask = build_opsd_prompts(
        tokenizer,
        env,
        problems,
        opsd_contexts,
        opsd_prompt_length,
    )
    action_mask = (all_roles == int(TokenRole.ACTION)) & (all_action_idx == 0)
    action_toks_per = [all_tokens[i][action_mask[i]].tolist() for i in range(all_tokens.shape[0])]
    if audit:
        audit_opsd_inputs(
            tokenizer,
            problems,
            opsd_contexts,
            action_toks_per,
            model_name=audit_model_name,
            num_workers=audit_num_workers,
            verbose=verbose,
        )
    combined = [aug + action for aug, action in zip(aug_lists, action_toks_per)]
    opsd_tokens, _ = pad_and_collate(
        combined,
        pad_id=pad_id,
        force_length=opsd_seq_len,
        how='left',
        description='opsd teacher sequence',
        raise_on_truncation=not allow_prompt_truncation,
        verbose=verbose,
    )

    opsd_teacher_pos = np.zeros_like(all_tokens, dtype=np.int32)
    for i, (aug, action) in enumerate(zip(aug_lists, action_toks_per)):
        student_action_cols = np.where(action_mask[i])[0]
        n_action = min(len(action), len(student_action_cols))
        pad_len = max(opsd_seq_len - len(aug) - len(action), 0)
        teacher_action_cols = pad_len + len(aug) + np.arange(n_action, dtype=np.int32)
        opsd_teacher_pos[i, student_action_cols[:n_action]] = teacher_action_cols

    return opsd_tokens.astype(np.int32), opsd_teacher_pos.astype(np.int32), opsd_mask.astype(np.float32)


def audit_opsd_inputs(
    tokenizer,
    problems,
    opsd_contexts,
    action_toks_per,
    model_name='gemini-2.5-flash-lite',
    num_workers=16,
    verbose=False,
):
    """Gemini sanity-check for first-batch OPSD prompt/context/rollout structure."""
    api_key = os.environ['GEMINI_API_KEY']

    pc = jax.process_count()
    assert len(problems) % pc == 0, f'batch size {len(problems)} not divisible by process_count {pc}'
    per_host = len(problems) // pc
    local = np.arange(jax.process_index() * per_host, (jax.process_index() + 1) * per_host)
    local_problems = [str(problems[i]) for i in local]
    local_contexts = [str(opsd_contexts[i] or '') for i in local]
    local_rollouts = [tokenizer.decode(action_toks_per[i]) for i in local]
    prompts = [get_opsd_audit_user_prompt(p, c, r) for p, c, r in zip(local_problems, local_contexts, local_rollouts)]

    run_audit(
        prompts,
        lambda audit_prompts: gemini_parallel(
            audit_prompts,
            OPSD_AUDIT_SYSTEM_PROMPT,
            api_key,
            model=model_name,
            num_workers=num_workers,
            desc='OPSD audit',
            verbose=verbose,
        )[0],
        source='OPSD',
    )


def align_base_teacher_hidden(train_state, base_params, opsd_tokens, opsd_teacher_pos, pad_id):
    """Run the frozen base model on teacher-conditioned sequences and align hidden states to student columns."""
    teacher_params = jax.tree.map(jax.lax.stop_gradient, base_params)
    opsd_token_mask = jnp.where(opsd_tokens != pad_id, 1, 0).astype(jnp.int32)
    teacher_hidden, _ = train_state.call_model(
        opsd_tokens, opsd_token_mask, cache=None, params=teacher_params, return_hidden=True
    )
    teacher_pred_pos = jnp.clip(opsd_teacher_pos[:, 1:] - 1, 0, teacher_hidden.shape[1] - 1)
    teacher_hidden_for_student = jnp.take_along_axis(teacher_hidden, teacher_pred_pos[..., None], axis=1)
    teacher_hidden_for_student = jnp.concatenate(
        [teacher_hidden_for_student, jnp.zeros_like(teacher_hidden_for_student[:, :1])],
        axis=1,
    )
    return teacher_hidden_for_student, teacher_params['Dense_0']['kernel']


def teacher_topk(teacher_hidden, teacher_lm_kernel, topk=100, logits_sharding=None):
    """Teacher top-k logprobs/indices for a hidden chunk."""
    assert topk > 0, f'OPSD distillation_topk must be > 0, got {topk}'
    teacher_logits = (teacher_hidden @ teacher_lm_kernel).astype(jnp.float32)
    if logits_sharding is not None:
        teacher_logits = jax.lax.with_sharding_constraint(teacher_logits, logits_sharding)
    teacher_log_norm = jax.nn.logsumexp(teacher_logits, axis=-1, keepdims=True)
    teacher_topk_logits, teacher_topk_idx = jax.lax.top_k(teacher_logits, topk)
    teacher_topk_logprobs = jax.lax.stop_gradient(teacher_topk_logits - teacher_log_norm)
    return teacher_topk_logprobs, teacher_topk_idx


def token_divergence(student_logits, student_log_norm, teacher_topk_logprobs, teacher_topk_idx, alpha, add_tail=True):
    """Tokenwise KL/JSD-style divergence on the base teacher's top-k support."""
    student_topk_logits = jnp.take_along_axis(student_logits, teacher_topk_idx, axis=-1)
    student_topk_logprobs = student_topk_logits - student_log_norm[..., None]

    def _add_tail(logprobs):
        log_s = jnp.clip(jax.nn.logsumexp(logprobs, axis=-1, keepdims=True), max=-1e-7)
        tail = jnp.log(-jnp.expm1(log_s))
        return jnp.concatenate([logprobs, tail], axis=-1)

    if add_tail:
        student = _add_tail(student_topk_logprobs)
        teacher = _add_tail(teacher_topk_logprobs)
    else:
        student = student_topk_logprobs - jax.nn.logsumexp(student_topk_logprobs, axis=-1, keepdims=True)
        teacher = teacher_topk_logprobs - jax.nn.logsumexp(teacher_topk_logprobs, axis=-1, keepdims=True)

    kl_qp = lambda p_lp, q_lp: jnp.sum(jnp.exp(q_lp) * (q_lp - p_lp), axis=-1)
    if alpha == 0.0:
        return kl_qp(student, teacher)
    if alpha == 1.0:
        return kl_qp(teacher, student)
    mixed = jax.nn.logsumexp(
        jnp.stack([student + jnp.log(1.0 - alpha), teacher + jnp.log(alpha)], axis=0),
        axis=0,
    )
    return (1.0 - alpha) * kl_qp(mixed, student) + alpha * kl_qp(mixed, teacher)


def loss_opsd(
    opsd_kl_token,
    token_logprobs,
    inference_logprobs,
    action_mask,
    opsd_mask,
    is_clip,
):
    if is_clip > 0:
        is_ratio = jnp.clip(
            jnp.exp(jnp.clip(jax.lax.stop_gradient(token_logprobs - inference_logprobs), -20.0, 20.0)),
            max=is_clip,
        )
        opsd_kl_token = opsd_kl_token * is_ratio

    opsd_token_mask = action_mask.astype(jnp.float32) * opsd_mask[:, None].astype(jnp.float32)
    return jnp.sum(opsd_kl_token * opsd_token_mask) / (jnp.sum(opsd_token_mask) + 1e-8)
