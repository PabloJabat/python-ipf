import numpy as np


class IPF:
    def __init__(
        self,
        target_marginals_0: np.ndarray,
        target_marginals_1: np.ndarray,
        iterations: int = 50,
    ) -> None:
        self.target_marginals_0 = target_marginals_0
        self.target_marginals_1 = target_marginals_1
        self.iterations = iterations

    def run(self, seed: np.ndarray) -> np.ndarray:
        """Run IPF"""

        result = seed

        for _ in range(self.iterations):
            result_0 = _iterate_axis(result, self.target_marginals_0, 0)
            result_1 = _iterate_axis(result_0, self.target_marginals_1, 1)

            result = result_1

        return result


def _iterate_axis(
    data: np.ndarray, target_marginals: np.ndarray, axis: int
) -> np.ndarray:
    """Make an interation on on axis"""

    factor = target_marginals / _compute_marginals(data, axis)

    return _multiply(data, factor, axis=axis)


def _multiply(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Multiply an array by another one dimensional array across a given axis"""

    assert b.ndim == 1

    dim_array = np.ones((1, a.ndim), int).ravel()
    dim_array[0 if axis else 1] = -1

    return a * b.reshape(dim_array)


def _compute_marginals(data: np.ndarray, axis: int) -> np.ndarray:
    """Compute the marginal count on one axis"""

    return data.sum(axis=axis)
