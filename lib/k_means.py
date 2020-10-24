import numpy as np

def initialize_prototypes(X, M):
    N = X.shape[0]
    mean = X.mean(axis=1).reshape(-1, 1)
    std = X.std(axis=1).reshape(-1, 1)

    variations = []
    for n in np.arange(N):
        variations.append(np.random.uniform(-std[n], std[n], M))
    variations = np.array(variations)

    prototypes = np.tile(mean, M)
    prototypes += variations
    return prototypes


def dist(w_q, x):
    return np.linalg.norm(w_q - x)
