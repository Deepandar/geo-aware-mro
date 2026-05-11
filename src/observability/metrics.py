import time

from functools import wraps


def track_latency(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):

        start = time.time()

        result = fn(*args, **kwargs)

        end = time.time()

        latency = round(end - start, 4)

        print(f"[METRIC] {fn.__name__} latency={latency}s")

        return result

    return wrapper
