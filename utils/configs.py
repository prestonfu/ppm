import ast

from dotenv import load_dotenv
from omegaconf import OmegaConf


def to_config(obj):
    return OmegaConf.create(obj)


def to_plain(obj):
    return OmegaConf.to_container(obj, resolve=True) if OmegaConf.is_config(obj) else obj


def load_config(defaults, argv):
    load_dotenv()
    return OmegaConf.merge(to_config(defaults), OmegaConf.from_dotlist(argv[1:])), {}


def conf_default_to(x, default):
    return default if x == -1 else x


def parse_wallclock_interval(value):
    hours, minutes, seconds = (int(part) for part in value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def normalize_env_config(env_cfg):
    from lmpo.envs.env_creator import ENV_REGISTRY, get_env_defaults

    env_cfg = dict(to_plain(env_cfg))
    env_name = env_cfg.pop('env_name', '')
    if not env_name:
        return to_config(env_cfg)
    assert env_name in ENV_REGISTRY, env_name

    meta = {key: env_cfg.pop(key) for key in ('env_nickname', 'num_epochs') if key in env_cfg}
    overrides = dict(to_plain(env_cfg.pop('overrides', {})))
    if env_name != 'multitask':
        return to_config({**get_env_defaults(env_name), **overrides, **env_cfg, **meta, 'env_name': env_name})

    env_list = env_cfg.pop('env_list', [])
    if isinstance(env_list, str) and env_list:
        env_list = ast.literal_eval(env_list)

    sub_envs = []
    for item in env_list:
        item = dict(to_plain(item))
        sub_name = item.pop('env_name')
        sub_overrides = dict(to_plain(item.pop('overrides', {})))
        sub_envs.append({**get_env_defaults(sub_name), **overrides, **sub_overrides, **item, 'env_name': sub_name})

    return to_config(
        {
            **get_env_defaults('multitask'),
            **env_cfg,
            **meta,
            **overrides,
            'env_name': 'multitask',
            'env_list': sub_envs,
        }
    )


def normalize_reward_config(config):
    from lmpo.core.metrics import reward_metric_defaults

    config = to_config(config)
    if 'reward' not in config:
        return config

    reward_cfg = config.reward
    metrics_cfg = dict(to_plain(reward_cfg.get('metrics', {})))
    weights = dict(to_plain(reward_cfg.get('weights', {})))
    metric_names = list(metrics_cfg)
    assert all(name in reward_metric_defaults for name in metric_names), metric_names
    assert all(name in weights for name in metric_names), metric_names
    assert all(name in metric_names or name not in reward_metric_defaults for name in weights), list(weights)

    context = {
        'model_dir': config.model_dir,
        'inference_batch_per_device': config.sampling.inference_batch_per_device,
        'fsdp': config.sampling.fsdp,
        'tp_size': config.sampling.tp_size,
    }
    active = {}
    for metric_name in metric_names:
        metric_cfg = {**reward_metric_defaults[metric_name], **to_plain(metrics_cfg[metric_name])}
        for field, default in context.items():
            if field == 'model_dir' and metric_cfg.get(field) == '':
                metric_cfg[field] = default
            elif metric_cfg.get(field) == -1:
                metric_cfg[field] = default
        active[metric_name] = metric_cfg

    reward_cfg.metrics = to_config(active)
    if 'diagnostic_interval' not in reward_cfg:
        reward_cfg.diagnostic_interval = int(config.get('diagnostic_every_rollout_iters', 50))
    return config
