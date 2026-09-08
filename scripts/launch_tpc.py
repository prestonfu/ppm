import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import PurePosixPath
from pathlib import Path

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


def worker_ips(config_path):
    output = subprocess.check_output(['tpc', 'ips', str(config_path)], text=True)
    return list(dict.fromkeys(re.findall(r'(?:\d{1,3}\.){3}\d{1,3}', output)))


def rsync_repo(local_path, remote_path, config_path, excludes):
    rsync_base = ['rsync', '-az', '--info=stats1']
    rsync_base.extend(f'--exclude={exclude}' for exclude in excludes)
    rsync_base.extend(['-e', 'ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR', local_path])
    for ip in worker_ips(config_path):
        run([*rsync_base, f'kvfrans@{ip}:{remote_path}'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--tmux-session', default='')
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--setup', default='')
    parser.add_argument('--upload-path', default='')
    parser.add_argument('--workdir', default='')
    parser.add_argument('--rsync-to', default='')
    parser.add_argument('--rsync-exclude', action='append', default=[])
    args = parser.parse_args()

    cmd = experiment_command(args.config, args.override)
    setup = f'{args.setup}\n' if args.setup else ''
    workdir = args.workdir or str(
        PurePosixPath(args.rsync_to.rstrip('/')).parent if args.rsync_to else Path.cwd().parent
    )
    launch_script = f'#!/usr/bin/env bash\nset -e\n{setup}cd {shlex.quote(workdir)}\n{cmd}\n'
    tpc_config = {
        'project': args.project,
        'zone': args.zone,
        'name': args.name,
        'launch_script': launch_script,
    }
    if args.tmux_session:
        tpc_config['tmux_session_name'] = args.tmux_session
    if args.upload_path:
        tpc_config['upload_path'] = args.upload_path
        tpc_config['upload_remove_remote'] = False
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write('configure_tpc(\n')
        for key, value in tpc_config.items():
            f.write(f'    {key}={value!r},\n')
        f.write(')\n')
        config_path = Path(f.name)

    action = 'upload+launch' if args.upload_path else 'launch'
    if args.rsync_to:
        excludes = [*DEFAULT_RSYNC_EXCLUDES, *args.rsync_exclude]
        rsync_repo(str(Path.cwd()) + '/', args.rsync_to, config_path, excludes)
    run(['tpc', action, str(config_path)])


if __name__ == '__main__':
    main()
