import time
from functools import wraps

def track_timing(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        duration = round(end - start, 4)

        print(f"[METRIC] {func.__name__} executed in {duration}s")

        return result

    return wrapper
