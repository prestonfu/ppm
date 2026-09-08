import os
import pickle
import shutil
import subprocess


def findmnt(path, field):
    r = subprocess.run(['findmnt', '-n', '-T', path, '-o', field], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def get_gcs_bucket(path):
    """Return 'gs://<bucket>' if path is on a gcsfuse mount, else None."""
    if not os.path.isabs(path) or findmnt(path, 'FSTYPE') != 'fuse.gcsfuse':
        return None
    return f'gs://{findmnt(path, "SOURCE")}'


def gcs_uri(path):
    return f'{get_gcs_bucket(path)}/{os.path.relpath(path, findmnt(path, "TARGET"))}'


def gcloud(*args):
    return subprocess.run(['gcloud', 'storage', *args], capture_output=True, text=True)


def smart_open(path, mode):
    """Open binary read/write. Streams directly from gs:// on gcsfuse mounts."""
    if get_gcs_bucket(path):
        import tensorflow as tf

        return tf.io.gfile.GFile(gcs_uri(path), mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode)


def pkl_save(obj, path):
    with smart_open(path, 'wb') as f:
        pickle.dump(obj, f)


def file_size(path):
    if get_gcs_bucket(path):
        import tensorflow as tf

        return tf.io.gfile.stat(gcs_uri(path)).length
    return os.path.getsize(path)


def pkl_load(path, use_tqdm=False):
    if not use_tqdm:
        with smart_open(path, 'rb') as f:
            return pickle.load(f)
    from tqdm import tqdm

    total = file_size(path)
    chunks = []
    with smart_open(path, 'rb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(path)) as bar:
        while True:
            buf = f.read(64 * 1024 * 1024)
            if not buf:
                break
            chunks.append(buf)
            bar.update(len(buf))
    return pickle.loads(b''.join(chunks))


def file_exists(path):
    if not get_gcs_bucket(path):
        return os.path.exists(path)
    return gcloud('ls', gcs_uri(path)).returncode == 0


def list_dir(path):
    if not get_gcs_bucket(path):
        return os.listdir(path) if os.path.isdir(path) else []
    uri = gcs_uri(path).rstrip('/') + '/'
    r = gcloud('ls', uri)
    if r.returncode != 0:
        return []
    return [line.rstrip('/').rsplit('/', 1)[-1] for line in r.stdout.split() if line != uri]


def rm_dir(path):
    if not get_gcs_bucket(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        return
    gcloud('rm', '-r', gcs_uri(path).rstrip('/') + '/')
