import importlib.util
import shlex
from pathlib import Path


def load_experiment(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ENTRYPOINT, dict(module.CONFIG)


def format_value(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return str(value)


def parse_overrides(values):
    overrides = {}
    for item in values or []:
        key, value = item.split('=', 1)
        overrides[key] = value
    return overrides


def experiment_command(config_path, overrides=None):
    entrypoint, config = load_experiment(config_path)
    config.update(parse_overrides(overrides))
    args = ['python', '-m', entrypoint]
    args.extend(f'{key}={format_value(value)}' for key, value in sorted(config.items()))
    return shlex.join(args)
