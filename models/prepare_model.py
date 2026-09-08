import argparse
import json
import os
import shutil
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from lmpo.models.qwen3 import create_model_from_hf
from lmpo.utils.checkpoint import Checkpoint


load_dotenv()


def copytree(src, dst):
    if tf.io.gfile.exists(dst):
        tf.io.gfile.rmtree(dst)
    tf.io.gfile.makedirs(dst)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = dst if rel == '.' else f'{dst.rstrip("/")}/{rel}'
        tf.io.gfile.makedirs(out_dir)
        for file in files:
            tf.io.gfile.copy(os.path.join(root, file), f'{out_dir}/{file}', overwrite=True)


parser = argparse.ArgumentParser(description='Download a HF model and convert it to an LMPO params.pkl model dir.')
parser.add_argument('--model_id', required=True)
parser.add_argument('--model_dir', required=True)
parser.add_argument('--local_tmp', default='/tmp/lmpo-models')
args = parser.parse_args()

model_dir = args.model_dir if args.model_dir.startswith('gs://') else os.path.expanduser(args.model_dir)
success = f'{model_dir.rstrip("/")}/_SUCCESS'
if tf.io.gfile.exists(success):
    print(f'{model_dir} already has _SUCCESS; skipping.')
    raise SystemExit

name = args.model_id.strip('/').replace('/', '--')
tmp = Path(args.local_tmp).expanduser().absolute()
hf_dir, work_dir = tmp / 'hf' / name, tmp / 'converted' / name
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True)

snapshot_download(repo_id=args.model_id, local_dir=hf_dir)
_, params = create_model_from_hf(str(hf_dir))
ckpt = Checkpoint(str(work_dir / 'params.pkl'))
ckpt.save({'params': params})
for file in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
    shutil.copy(hf_dir / file, work_dir / file)
(work_dir / 'model_info.json').write_text(json.dumps({'source': str(hf_dir), 'format': 'params.pkl'}, indent=2) + '\n')
copytree(work_dir, model_dir)
with tf.io.gfile.GFile(success, 'w') as f:
    f.write('ok\n')
