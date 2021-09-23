import numpy as np

from . import utils

if __name__ == "__main__":

    target_marginals_0 = np.array([35, 40, 25])
    target_marginals_1 = np.array([20, 30, 35, 15])
    seed = np.array([[6, 6, 3], [8, 10, 10], [9, 10, 9], [3, 14, 8]])

    result = seed
    for _ in range(100):
        new_result = utils.iterate_axis(result, target_marginals_1, 1)
        new_result = utils.iterate_axis(new_result, target_marginals_0, 0)

        result = new_result

    print(result)

    print("Done ...")
