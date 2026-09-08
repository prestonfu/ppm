import dataclasses
import os

import jax
import numpy as np

from lmpo.core.sampling import TokenRole
from lmpo.models.qwen3 import create_model_from_ckpt, get_tp_spec
from lmpo.utils.configs import to_plain
from lmpo.utils.sharding import create_sharding, get_shard_params_fn
from lmpo.utils.timer import Timer

from .common import MetricHelper, ModelBundle, combine_turn_results
from .process_gemini import ProcessGeminiMetric
from .rubric_gemini import RubricGeminiMetric
from .rubric_qwen import RubricQwenMetric
from .verifree import VerifreeMetric


class RewardMetrics(
    VerifreeMetric,
    RubricQwenMetric,
    RubricGeminiMetric,
    ProcessGeminiMetric,
    MetricHelper,
):
    GEMINI_METRICS = ('rubric_score_gemini', 'process_rewards_gemini')
    # Per-metric state-derived kwargs: kwarg -> (single_attr, per_turn_attr_or_None).
    STATE_KWARGS = {
        'verifree': {'oracle_solutions': ('oracle_solution', None)},
        'rubric_score_qwen': {
            'problems': ('problem', 'per_turn_problems'),
            'rubrics': ('rubric', 'per_turn_rubrics'),
            'rubric_n_items': ('rubric_n_items', None),
            'makes_obsoletes': ('makes_obsolete', None),
        },
        'rubric_score_gemini': {
            'problems': ('problem', 'per_turn_problems'),
            'rubrics': ('rubric', 'per_turn_rubrics'),
            'rubric_n_items': ('rubric_n_items', None),
            'makes_obsoletes': ('makes_obsolete', None),
        },
        'process_rewards_gemini': {'problems': ('problem', 'per_turn_problems')},
    }

    def __init__(
        self,
        config,
        env,
        tokenizer,
        model,
        params_shard,
        sampler,
        data_shard,
        no_shard,
        get_local_slice,
        shard_data_fn,
    ):
        reward_config = config.reward
        reward_configs = reward_config.metrics
        self.reward_names = list(reward_configs)
        self.tokenizer = tokenizer
        self.raise_on_truncation = not config.allow_prompt_truncation
        self.prompt_length = int(config.env.prompt_length)
        self.tokens_per_action = int(config.env.tokens_per_action)
        self.num_turns = int(env.num_turns)
        self.env_enable_thinking = int(config.env.enable_thinking)
        self.fns = {name: getattr(self, name) for name in self.reward_names}
        self.metric_configs = {name: to_plain(reward_configs.get(name, {})) for name in self.reward_names}
        self.train_metric_names = {name for name in self.reward_names if float(reward_config.weights.get(name, 0)) != 0}
        self.diagnostic_interval = int(reward_config.get('diagnostic_interval', 50))
        clean_path = lambda x: os.path.abspath(os.path.expanduser(x))
        sampler_key = (clean_path(config.model_dir), bool(config.sampling.fsdp), model.tp_size)
        self.policy_bundle = ModelBundle(
            model=model,
            shard_params=get_shard_params_fn(params_shard, no_shard),
            data_shard=data_shard,
            no_shard=no_shard,
            local_slice=get_local_slice,
            shard_data=shard_data_fn,
            sampler=sampler,
            uses_policy_params=True,
        )
        separate_bundles = {}
        self.bundle_by_metric = {}
        for metric in self.reward_names:
            if metric in self.GEMINI_METRICS:
                self.bundle_by_metric[metric] = None
                continue
            metric_cfg = reward_configs[metric]
            model_key = (clean_path(metric_cfg.model_dir), bool(metric_cfg.fsdp), int(metric_cfg.tp_size))
            if model_key == sampler_key:
                self.bundle_by_metric[metric] = self.policy_bundle
                continue
            if model_key not in separate_bundles:
                judge_model, judge_params = create_model_from_ckpt(model_key[0])
                params_shard_, no_shard_, data_shard_, shard_data_, local_slice_ = create_sharding(
                    judge_params,
                    fsdp=model_key[1],
                    tp_size=model_key[2],
                    get_tp_spec=get_tp_spec,
                )
                separate_bundles[model_key] = ModelBundle(
                    model=dataclasses.replace(judge_model, mesh=data_shard_.mesh, tp_size=model_key[2]),
                    cpu_params=jax.tree.map(np.asarray, judge_params),
                    shard_params=get_shard_params_fn(params_shard_, no_shard_),
                    data_shard=data_shard_,
                    no_shard=no_shard_,
                    local_slice=local_slice_,
                    shard_data=shard_data_,
                    sampler=None,
                )
            self.bundle_by_metric[metric] = separate_bundles[model_key]
        self.has_separate_model = bool(separate_bundles)

    def state_kwargs(self, states, turn=None):
        """Collect state-derived kwargs for all active metrics."""
        if not states:
            return {}
        first = states[0]
        out = {}
        for metric in self.reward_names:
            for kwarg, (single, per_turn) in self.STATE_KWARGS.get(metric, {}).items():
                if per_turn and turn is not None and hasattr(first, per_turn):
                    vals = [getattr(s, per_turn)[turn] for s in states]
                elif hasattr(first, single):
                    vals = [getattr(s, single) for s in states]
                else:
                    continue
                out[kwarg] = vals
        return out

    def __bool__(self):
        return bool(self.reward_names)

    def __repr__(self):
        return f'RewardMetrics({self.reward_names})'

    def evaluate_rollout(
        self,
        rollout,
        base_params,
        sampling_params,
        sampler,
        rng,
        *,
        num_rollouts,
        total_rollout_iters,
        prompt_length,
        tokens_per_action,
        run_all_metrics=False,
        audit=False,
        verbose=False,
    ):
        if not self.reward_names:
            return {}, {}, rng
        if self.has_separate_model and sampling_params is not None:
            raise ValueError('release policy sampling parameters before loading a separate metric model')

        n = int(num_rollouts)
        tokens = rollout.tokens[:n]
        roles = rollout.roles[:n]
        states = rollout.env_states[:n]
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        due = [
            name
            for name in self.reward_names
            if run_all_metrics
            or name in self.train_metric_names
            or (self.diagnostic_interval > 0 and total_rollout_iters % self.diagnostic_interval == 0)
        ]
        turn_results = []
        renders = {}
        active_bundle = None

        try:
            for turn in range(self.num_turns):
                rng, reward_key = jax.random.split(rng)
                turn_result = {}
                common = dict(
                    tokens=tokens,
                    action_mask=((roles == int(TokenRole.ACTION)) & (rollout.action_idx[:n] == turn)),
                    prompt_mask=roles == int(TokenRole.OBSERVATION),
                    prompt_length=prompt_length,
                    tokens_per_action=tokens_per_action,
                    num_turns=self.num_turns,
                    env_enable_thinking=self.env_enable_thinking,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    rng=reward_key,
                    verbose=verbose,
                    audit=audit and turn == 0,
                    **self.state_kwargs(states, turn=turn),
                )
                for name in due:
                    bundle = self.bundle_by_metric[name]
                    if bundle is not active_bundle:
                        if active_bundle is not None:
                            active_bundle.release()
                        active_bundle = bundle
                        if active_bundle is not None:
                            active_bundle.load(self.tokenizer, base_params, sampling_params, sampler)
                    kwargs = {**self.metric_configs[name], **common, 'bundle': active_bundle}
                    timer = Timer(f'reward_metrics/{name}')
                    timer.start()
                    try:
                        value, extras, metric_renders = self.fns[name](**kwargs)
                    finally:
                        timer.end()
                    turn_result[name] = value
                    turn_result.update({f'{name}/{key}': item for key, item in extras.items()})
                    for key, item in metric_renders.items():
                        render_name = f'turn{turn}/{name}/{key}' if self.num_turns > 1 else f'{name}/{key}'
                        renders[render_name] = item
                turn_results.append(turn_result)
        finally:
            if active_bundle is not None:
                active_bundle.release()

        return combine_turn_results(turn_results) if due else {}, renders, rng
