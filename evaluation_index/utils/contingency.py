import numpy as np


def _coerce_1d_labels(labels, name):
    """Convert an input label sequence to a 1D numpy array."""
    arr = np.asarray(labels)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"{name} must be a 1D sequence of labels")

    return arr


def validate_label_inputs(labels1, labels2):
    """Validate and normalize two label sequences."""
    labels1 = _coerce_1d_labels(labels1, "labels1")
    labels2 = _coerce_1d_labels(labels2, "labels2")

    if labels1.shape[0] != labels2.shape[0]:
        raise ValueError(
            "labels1 and labels2 must have the same length "
            f"(got {labels1.shape[0]} and {labels2.shape[0]})"
        )
    if labels1.shape[0] == 0:
        raise ValueError("labels1 and labels2 must be non-empty")

    return labels1, labels2


def _unique_in_order(labels):
    return list(dict.fromkeys(labels.tolist()))


def contingency_table(labels1, labels2):
    """
    Calculate the contingency table between two clusterings.

    Parameters:
    labels1: List or array of labels for the first clustering (length N).
    labels2: List or array of labels for the second clustering (length N).

    Returns:
    n_ij: Contingency matrix of shape (K_A, K_B).
    """
    labels1, labels2 = validate_label_inputs(labels1, labels2)
    N = labels1.shape[0]

    unique_labels1 = _unique_in_order(labels1)
    unique_labels2 = _unique_in_order(labels2)

    K_A = len(unique_labels1)
    K_B = len(unique_labels2)

    label_to_idx1 = {label: i for i, label in enumerate(unique_labels1)}
    label_to_idx2 = {label: i for i, label in enumerate(unique_labels2)}

    n_ij = np.zeros((K_A, K_B), dtype=int)

    for i in range(N):
        idx1 = label_to_idx1[labels1[i].item() if hasattr(labels1[i], "item") else labels1[i]]
        idx2 = label_to_idx2[labels2[i].item() if hasattr(labels2[i], "item") else labels2[i]]
        n_ij[idx1, idx2] += 1

    return n_ij


def marginal_sums(n_ij):
    """
    a_i: Size of each cluster in the first clustering
    b_j: Size of each cluster in the second clustering
    N: Total number of elements.
    """
    a_i = n_ij.sum(axis=1)
    b_j = n_ij.sum(axis=0)
    N = n_ij.sum()
    return a_i, b_j, N


def partitions_equivalent(labels1, labels2):
    """Return True when two labelings induce the same partition."""
    n_ij = contingency_table(labels1, labels2)
    row_nnz = np.count_nonzero(n_ij, axis=1)
    col_nnz = np.count_nonzero(n_ij, axis=0)
    return bool(np.all(row_nnz <= 1) and np.all(col_nnz <= 1))


def comb(n, k):
    """Calculate combinations"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result


def combpairs(n_ij, a_i, b_j, N):
    """
    Calculate the number of pairs for clustering evaluation indices
    Parameters:
    n_ij: Contingency table matrix.
    a_i: Size of each cluster in the first clustering.
    b_j: Size of each cluster in the second clustering.
    N: Total number of elements.
    返回:
    N11: Number of pairs that are in the same cluster in both clusterings.
    N10: Number of pairs in the same cluster in C1 but different in C2.
    N01: Number of pairs in different clusters in C1 but same in C2.
    N00: Number of pairs in different clusters in both clusterings.
    """
    total_pairs = comb(N, 2)
    N11 = 0
    for i in range(n_ij.shape[0]):
        for j in range(n_ij.shape[1]):
            N11 += comb(n_ij[i, j], 2)
    sum_ai2 = sum(comb(a, 2) for a in a_i)
    sum_bj2 = sum(comb(b, 2) for b in b_j)
    N10 = sum_ai2 - N11
    N01 = sum_bj2 - N11
    N00 = total_pairs - N11 - N10 - N01
    return N11, N10, N01, N00
