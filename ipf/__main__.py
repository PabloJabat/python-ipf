import numpy as np

from .algorithm import IPF

if __name__ == "__main__":

    target_marginals_0 = np.array([35, 40, 25])
    target_marginals_1 = np.array([20, 30, 35, 15])
    seed = np.array([[6, 6, 3], [8, 10, 10], [9, 10, 9], [3, 14, 8]])

    ipf = IPF(target_marginals_0, target_marginals_1)

    print(ipf.run(seed))

    print("Done ...")
