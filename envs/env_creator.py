from lmpo.envs.poem_length import PoemLengthEnv, config as poem_config
from lmpo.envs.gsm8k import GSM8KEnv, config as gsm8k_config
from lmpo.envs.countdown import CountdownEnv, config as countdown_config
from lmpo.envs.multicountdown import MultiCountdownEnv, config as multicountdown_config
from lmpo.envs.deepscaler import DeepscalerEnv, config as deepscaler_config
from lmpo.envs.dapo_math_17k import DAPOMath17KEnv, config as dapo_math_17k_config
from lmpo.envs.pope_hard import PopeHardEnv, config as pope_hard_config
from lmpo.envs.polaris_acemath import PolarisAcemathEnv, config as polaris_acemath_config
from lmpo.envs.aime import Aime2024Env, Aime2025Env, aime2024_config, aime2025_config
from lmpo.envs.hmmt import Hmmt2025Env, config as hmmt2025_config
from lmpo.envs.manipulate_matrix import ManipulateMatrixEnv, config as manipulate_matrix_config
from lmpo.envs.multi_manipulate_matrix import MultiManipulateMatrixEnv, config as multi_manipulate_matrix_config
from lmpo.envs.gsm_infinite_hard import GsmInfiniteHardEnv, config as gsm_infinite_hard_config
from lmpo.envs.multitask import MultiTaskEnv, config as multitask_config


BASE_ENV_CONFIG = {
    'tokens_per_action': 32,
    'force_answer_at': -1,
    'force_end_think_at': -1,
    'num_tasks': -1,
    'num_turns': 1,
    'env_nickname': '',
    'obs_length': -1,
}


# Each entry is (EnvClass, default_config) or (EnvClass, default_config, extra_kwargs).
# extra_kwargs are passed to the constructor (e.g. {'train': False}).
ENV_REGISTRY = {
    'poem': (PoemLengthEnv, poem_config),
    'gsm8k': (GSM8KEnv, gsm8k_config),
    'gsm8k-test': (GSM8KEnv, gsm8k_config, {'train': False}),
    'countdown': (CountdownEnv, countdown_config),
    'multicountdown': (MultiCountdownEnv, multicountdown_config),
    'deepscaler': (DeepscalerEnv, deepscaler_config),
    'dapo_math_17k': (DAPOMath17KEnv, dapo_math_17k_config),
    'pope_hard': (PopeHardEnv, pope_hard_config),
    'polaris_acemath': (PolarisAcemathEnv, polaris_acemath_config),
    'aime2024': (Aime2024Env, aime2024_config),
    'aime2025': (Aime2025Env, aime2025_config),
    'hmmt2025': (Hmmt2025Env, hmmt2025_config),
    'manipulate_matrix': (ManipulateMatrixEnv, manipulate_matrix_config),
    'multi_manipulate_matrix': (MultiManipulateMatrixEnv, multi_manipulate_matrix_config),
    'gsm_infinite_hard': (GsmInfiniteHardEnv, gsm_infinite_hard_config),
    'multitask': (MultiTaskEnv, multitask_config),
}


def create_one_env(env_name, tokenizer, **kwargs):
    assert env_name != 'multitask'
    if env_name not in ENV_REGISTRY:
        raise ValueError(f'Unknown environment name: {env_name}')

    entry = ENV_REGISTRY[env_name]
    env_cls = entry[0]
    extra_kwargs = entry[2] if len(entry) > 2 else {}
    return env_cls(tokenizer, **{**extra_kwargs, **kwargs})


def get_env_defaults(env_name=None):
    if env_name is not None:
        return {**BASE_ENV_CONFIG, **ENV_REGISTRY[env_name][1]}
    return {name: get_env_defaults(name) for name in ENV_REGISTRY}


env_defaults = get_env_defaults()


def create_env(env_config, tokenizer):
    """Create one selected env or a multitask env from a normalized env config."""
    from lmpo.utils.configs import to_plain

    env_name = env_config.env_name
    if env_name != 'multitask':
        kwargs = to_plain(env_config)
        kwargs.pop('env_name', None)
        kwargs.pop('num_epochs', None)
        env = create_one_env(env_name, tokenizer, **kwargs)
        env_config.env_nickname = env.env_nickname
        return env

    sub_env_configs = []
    weights = []
    for conf in env_config.env_list:
        conf = to_plain(conf)
        weight = conf.pop('weight', 1)
        weights.append(weight)
        sub_env_configs.append(conf)

    envs_list = []
    for conf in sub_env_configs:
        sub_env_name = conf.pop('env_name')
        conf.pop('num_epochs', None)
        envs_list.append(create_one_env(sub_env_name, tokenizer, **conf))

    env = MultiTaskEnv(
        envs_list,
        env_nickname=env_config.env_nickname,
        weights=weights,
        shuffle_seed=env_config.get('shuffle_seed', 0),
    )
    probs = env.get_mixture()
    for conf, prob in zip(sub_env_configs, probs):
        conf['weight'] = float(prob)

    env_config.env_nickname = env.env_nickname
    return env
