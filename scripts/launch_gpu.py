import argparse
import shlex
import subprocess
from pathlib import PurePosixPath, Path

from launch_common import experiment_command

DEFAULT_RSYNC_EXCLUDES = [
    '.git/',
    '.ruff_cache/',
    '__pycache__/',
    '*.pyc',
    'wandb/',
    'outputs/',
    'checkpoints/',
    'datasets/',
    '*.npz',
]


def run(cmd):
    print(shlex.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--tmux-session', required=True)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--setup', default='')
    parser.add_argument('--workdir', default='')
    parser.add_argument('--rsync-to', required=True)
    parser.add_argument('--rsync-exclude', action='append', default=[])
    args = parser.parse_args()

    excludes = [*DEFAULT_RSYNC_EXCLUDES, *args.rsync_exclude]
    rsync_cmd = ['rsync', '-az', '--info=stats1']
    rsync_cmd.extend(f'--exclude={exclude}' for exclude in excludes)
    rsync_cmd.extend([str(Path.cwd()) + '/', f'{args.host}:{args.rsync_to}'])
    run(rsync_cmd)

    workdir = args.workdir or str(PurePosixPath(args.rsync_to.rstrip('/')).parent)
    setup = f'{args.setup}\n' if args.setup else ''
    launch_script = f'set -e\ncd {shlex.quote(workdir)}\n{setup}{experiment_command(args.config, args.override)}'
    remote_cmd = f'tmux new-session -d -s {shlex.quote(args.tmux_session)} {shlex.quote(launch_script)}'
    run(['ssh', args.host, remote_cmd])


if __name__ == '__main__':
    main()
