import time


class Timer:
    _records = {}
    _active = {}

    def __init__(self, name):
        self.name = name

    def start(self):
        if self.name in Timer._active:
            raise RuntimeError(f"Timer '{self.name}' already started.")
        Timer._active[self.name] = time.perf_counter()

    def end(self):
        if self.name not in Timer._active:
            raise RuntimeError(f"Timer '{self.name}' was not started.")
        start_time = Timer._active.pop(self.name)
        end_time = time.perf_counter()
        Timer._records.setdefault(self.name, []).append((start_time, end_time))

    @classmethod
    def get_last(cls, name):
        recs = cls._records.get(name)
        if not recs:
            return None
        start, end = recs[-1]
        return end - start

    @classmethod
    def times(cls):
        return {name: sum([end - start for start, end in recs]) for name, recs in cls._records.items()}

    @classmethod
    def reset(cls, name=None):
        if name is None:
            cls._records.clear()
            cls._active.clear()
        else:
            cls._records.pop(name, None)
            cls._active.pop(name, None)
