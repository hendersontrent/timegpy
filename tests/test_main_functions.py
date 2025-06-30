import numpy as np
from timegpy.gp import evolve
from timegpy.evaluate_expression import evaluate

#----------------- Test 1: Overall function works ------------------

# Simulate Gaussian noise and AR(1) data

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

# Run main function

df_all, df_best = evolve(X, y, n_procs=5)

#----------------- Test 2: Fitness calculation ------------------

# Simulate small number of just AR(1) data

def AR1(N, T, phi, seed=None):
    if seed is not None:
        np.random.seed(seed)

    ar1_data = np.zeros((N, T))
    for i in range(N):
        noise = np.random.normal(0, 1, T)
        ar1 = np.zeros(T)
        ar1[0] = noise[0]
        for t in range(1, T):
            ar1[t] = phi * ar1[t - 1] + noise[t]
        ar1_data[i] = ar1

    return ar1_data

x = AR1(N=2, T=100, phi=0.8, seed=123)

# Calculate "mean(X_t+0 * X_t+1)" feature manually

values = []
for i in range(99):
    values.append(x[0, i] * x[0, i+1])

result1 = np.nanmean(values)

# Implement timegpy's feature calculation

result2 = evaluate("mean(X_t+0 * X_t+1)", x, z_score=False)

# Test if the two are the same

result1 == result2[0]
