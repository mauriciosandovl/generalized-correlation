import time
import numpy as np
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        kwargs = "{kwargs} " if f"{kwargs}" != "{}" else ""
        print(
            f"Function {func.__name__}{args} " + kwargs + f"Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper


def get_norm(data):
    """Function to evaluate the euclidean norm of the position of the atoms and
    reduce dimensionality"""
    norm = np.array([np.linalg.norm(i) for j in data for i in j])
    norm = norm.reshape((data.shape[0], data.shape[1], 1))

    return norm


def gen_corr_coef(data, dim=3):
    """Computes generalized correlation coeficient"""
    corr = (1 - np.exp((-2*data) / dim)) ** 0.5

    return corr

