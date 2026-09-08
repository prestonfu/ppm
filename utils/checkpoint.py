import os
import pickle
import shutil
import time


def format_age(seconds):
    seconds = int(seconds)
    parts = []
    for suffix, unit_seconds in [
        ('mo', 30 * 24 * 60 * 60),
        ('d', 24 * 60 * 60),
        ('h', 60 * 60),
        ('m', 60),
        ('s', 1),
    ]:
        value, seconds = divmod(seconds, unit_seconds)
        if value or parts or suffix == 's':
            parts.append(f'{value}{suffix}')
    return ' '.join(parts)


class Checkpoint:
    def __init__(self, filename):
        self.filename = filename

    def save(self, data):
        data = dict(data)
        data['_timestamp'] = time.time()
        print(f'Writing checkpoint: {self.filename}')
        if self.filename.startswith('gs://'):
            import tensorflow as tf

            tf.io.gfile.makedirs(self.filename.rsplit('/', 1)[0])
            with tf.io.gfile.GFile(self.filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            tmp = self.filename + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(tmp, self.filename)
        print(f'Wrote checkpoint: {self.filename}')

    def load_as_dict(self):
        if self.filename.startswith('gs://'):
            import tensorflow as tf

            with tf.io.gfile.GFile(self.filename, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
        age = time.time() - data['_timestamp']
        print(f'Loaded {self.filename} (saved {format_age(age)} ago).')
        return data
