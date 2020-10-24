import numpy as np
from lib.k_means import initialize_prototypes, dist


def h_gauss(q, p, sigma):
    return np.exp((-dist(q, p)**2) / (2 * sigma**2))


def som_online(X, M, w0, epsilon0, sigma0, epsilon_plateau, decay_epsilon, decay_sigma):
    N, p = X.shape

    epsilon = epsilon0
    sigma = sigma0
    w = w0.copy()

    w_history = []
    epsilon_history = []
    sigma_history = []
    for i in np.arange(p):
        x = X[:,i]
        distances = [dist(w_q, x) for w_q in w.T]
        winning_neuron_index = np.argmin(distances)

        winning_neuron = w[:,winning_neuron_index]
        delta_w = np.array([
                    epsilon * h_gauss(q, winning_neuron_index, sigma) * (x - w[:,q])
                    for q in np.arange(M)
                  ]).T

        w += delta_w
        w_history.append(w.copy())
        if i > p * epsilon_plateau:
            epsilon *= decay_epsilon
        epsilon_history.append(epsilon)
        sigma *= decay_sigma
        sigma_history.append(sigma)
    return w, np.array(epsilon_history), np.array(sigma_history), np.array(w_history)


def som_online_2d(X, M, w0, epsilon0, sigma0, epsilon_plateau, decay_epsilon, decay_sigma):
    N, p = X.shape

    epsilon = epsilon0
    sigma = sigma0
    w = w0.copy()

    w_history = []
    epsilon_history = []
    sigma_history = []
    for i in np.arange(p):
        x = X[:,i]
        distances = np.zeros((M, M))
        for k in np.arange(M):
            for l in np.arange(M):
                w_k_l = w[:, k, l,]
                distances[k,l] = dist(w_k_l, x)

        winning_neuron_indices = np.unravel_index(distances.argmin(), distances.shape)

        winning_neuron = w[:,winning_neuron_indices[0], winning_neuron_indices[1]]
        delta_w = np.zeros((M, M, N))
        for k in np.arange(M):
            for l in np.arange(M):
                w_k_l = w[:,k,l]
                delta_w[k,l] = epsilon * h_gauss((k,l), winning_neuron_indices, sigma) * (x - w_k_l)

        w += delta_w
        w_history.append(w.copy())
        if i > p * epsilon_plateau:
            epsilon *= decay_epsilon
        epsilon_history.append(epsilon)
        sigma *= decay_sigma
        sigma_history.append(sigma)
    return w, np.array(epsilon_history), np.array(sigma_history), np.array(w_history)
