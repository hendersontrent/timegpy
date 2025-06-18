import numpy as np
from timegpy.gp import tsgp

#----------------- Simulate Gaussian noise and AR(1) data -----------------

def generate_ar1_vs_noise(N, T, phi, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # AR(1) samples
    ar1_data = np.zeros((N, T))
    for i in range(N):
        noise = np.random.normal(0, 1, T)
        ar1 = np.zeros(T)
        ar1[0] = noise[0]
        for t in range(1, T):
            ar1[t] = phi * ar1[t - 1] + noise[t]
        ar1_data[i] = ar1

    # Gaussian noise samples
    noise_data = np.random.normal(0, 1, (N, T))

    # Combine
    X = np.vstack([ar1_data, noise_data])

    # Labels
    y = np.array([1] * N + [0] * N)

    return X, y

X, y = generate_ar1_vs_noise(N=100, T=100, phi=0.8, seed=123)

#----------------- Test core function -----------------

df_all, df_best = tsgp(X, y)
