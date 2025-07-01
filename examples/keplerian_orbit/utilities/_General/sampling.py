import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt


def test_sobol_samples(seed):
    # # 1) Pick a seed
    # seed = 42
    # 2a) If you’re on SciPy ≥ 1.7.0 and < 1.9.0:
    sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
    # 2b) If you’re on SciPy ≥ 1.9.0:
    # rng = np.random.default_rng(seed)
    # sobol = qmc.Sobol(d=4, scramble=True, rng=rng)

    # 3) Generate 2**10 = 1 024 Sobol points
    #    (you can also use .random(1024) but random_base2 is safer for powers of 2)
    samples_unit = sobol.random_base2(m=10)
    samples_np = np.array(samples_unit)
    return samples_np
    plt.scatter(samples_np[:,0], samples_np[:,1])
    plt.show()


if __name__ == '__main__':
    plt.figure()
    for i in range(5):
        samples_np = test_sobol_samples(i)
        plt.scatter(samples_np[:,0], samples_np[:,1])
    plt.show()