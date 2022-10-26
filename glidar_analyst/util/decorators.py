

import time


def my_timer(func):

    def wrapper(*args, **kwargs):

        t = time.process_time_ns()
        result = func(*args, **kwargs)
        print(f'Running "{func.__name__}" in {(time.process_time_ns() - t) * 1e-6:.1f} ms')
        return result

    return wrapper
