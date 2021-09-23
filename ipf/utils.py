import numpy as np
from pydantic import BaseModel  # pylint: disable=no-name-in-module

# class TargetMarginals(BaseModel):
#     """Target marginals in one axis"""

#     target_marginals: np.ndarray
#     axis: int


def multiply(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Multiply an array by another one dimensional array across a given axis"""

    assert b.ndim == 1

    dim_array = np.ones((1, a.ndim), int).ravel()
    dim_array[0 if axis else 1] = -1

    return a * b.reshape(dim_array)


def compute_marginals(data: np.ndarray, axis: int) -> np.ndarray:
    """Compute the marginal count on one axis"""

    return data.sum(axis=axis)


def iterate_axis(
    data: np.ndarray, target_marginals: np.ndarray, axis: int
) -> np.ndarray:
    """Make an interation on on axis"""

    factor = target_marginals / compute_marginals(data, axis)

    return multiply(data, factor, axis=axis)
