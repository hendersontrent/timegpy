import numpy as np
import timegpy

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

X, y = generate_ar1_vs_noise(N=1000, T=1000, phi=0.8, seed=123)

#----------------- Test core function -----------------

X, y, df_all, df_best = tsgp(
    X, y,
    pop_size=1000,
    n_generations=5,
    fitness_threshold=1,
    verbose=True,
    tournament_size=20,
    n_generation_improve=1
)