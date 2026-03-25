import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma


def PHSIC(X, Y, sX=None, sY=None, nrperm=0):
    """
    Calculates the Hilbert-Schmidt Independence Criterion (HSIC) between X and Y using RBF kernels.

    Args:
        X (np.ndarray): Nxd1 matrix of samples
        Y (np.ndarray): Nxd2 matrix of samples
        sX (float, optional): Kernel bandwidth for X. If None, automatically chosen.
        sY (float, optional): Kernel bandwidth for Y. If None, automatically chosen.
        nrperm (int): Number of permutations for p-value estimation (only 0 supported for gamma approx)

    Returns:
        float: PHSIC value (centered HSIC statistic)
    """
    if nrperm != 0:
        raise ValueError("Only nrperm=0 (gamma approximation) is supported in this implementation")

    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError("X and Y must have the same number of samples")

    # Estimate kernel bandwidths if not provided
    sX = guess_sigma(X) if sX is None else sX
    sY = guess_sigma(Y) if sY is None else sY

    # Calculate RBF kernel matrices
    KX = rbf_kernel(X, sX)
    KY = rbf_kernel(Y, sY)

    # Center the kernel matrices
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    KX_centered = H @ KX @ H
    KY_centered = H @ KY @ H

    # Calculate HSIC statistic
    hsic = np.trace(KX_centered @ KY) / (N ** 2)

    # Calculate mean under H0 for centering
    KX_sums = np.sum(KX, axis=1)
    KX_total = np.sum(KX_sums)
    KY_sums = np.sum(KY, axis=1)
    KY_total = np.sum(KY_sums)

    x_mu = (KX_total - N) / (N * (N - 1))
    y_mu = (KY_total - N) / (N * (N - 1))
    mean_H0 = (1.0 + x_mu * y_mu - x_mu - y_mu) / N

    return hsic - mean_H0


def rbf_kernel(X, sigma):
    """
    Computes the RBF kernel matrix for given data.

    Args:
        X (np.ndarray): Nxd matrix of samples
        sigma (float): Kernel bandwidth

    Returns:
        np.ndarray: NxN kernel matrix
    """
    pairwise_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


def guess_sigma(X, method=0):
    """
    Heuristically guesses a good kernel width sigma.

    Args:
        X (np.ndarray): Input data matrix
        method (int): Heuristic method to use (0=Arthur's, 1=old, 2=LOO)

    Returns:
        float: Estimated sigma value
    """
    if method == 0:
        # Arthur's heuristic
        norms = get_norms(X)
        upper_tri = np.triu(norms, k=1)
        non_zero = upper_tri[upper_tri > 0]
        return np.sqrt(0.5 * np.median(non_zero))
    elif method == 1:
        # Old heuristic
        norms = get_norms(X)
        return np.sqrt(0.5 * np.median(norms))
    elif method == 2:
        # LOO estimator (not implemented here for simplicity)
        raise NotImplementedError("LOO estimator method not implemented")
    else:
        raise ValueError("Invalid method specified")


def get_norms(X):
    """
    Computes pairwise squared Euclidean distances between samples.

    Args:
        X (np.ndarray): Input data matrix

    Returns:
        np.ndarray: Matrix of pairwise distances
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return cdist(X, X, 'sqeuclidean')