from evaluation_index.utils.contingency import *
import math

def entropy(labels1, labels2):
    """
        Calculate Shannon Entropy and Joint Shannon Entropy.

        Parameters:
        labels1: Labels of the first clustering.
        labels2: Labels of the second clustering.

        Returns:
        H_u:  Entropy of labels1.
        H_v:  Entropy of labels2.
        H_uv: Joint entropy of labels1 and labels2.
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    contingency = contingency_table(labels1, labels2)
    n = labels1.shape[0]
    # Calculate H(U)
    H_u = 0.0
    for ai in contingency.sum(axis=1):
        if ai > 0:
            p = ai / n
            H_u -= p * math.log(p)
    # Calculate H(V)
    H_v = 0.0
    for bj in contingency.sum(axis=0):
        if bj > 0:
            p = bj / n
            H_v -= p * math.log(p)
    # Calculate H(U,V)
    H_uv = 0.0
    for nij in contingency.flatten():
        if nij > 0:
            p = nij / n
            H_uv -= p * math.log(p)
    return H_u, H_v, H_uv

#Generalized Tsallis q-entropy
def tsallis_q_entropy(labels1, labels2, q):
    """
        Calculate Generalized Tsallis q-entropy.

        Parameters:
        labels1: Labels of the first clustering.
        labels2: Labels of the second clustering.
        q:       The entropic index (q parameter).

        Returns:
        H_uq:  Tsallis entropy of labels1.
        H_vq:  Tsallis entropy of labels2.
        H_uvq: Joint Tsallis entropy.
    """
    if abs(q - 1.0) < 1e-9:
        return entropy(labels1, labels2)
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    n = labels1.shape[0]
    # Calculate H_q(U) and H_q(V)
    values, counts = np.unique(labels1, return_counts=True)
    H_uq = (1.0 / (q - 1.0)) * (1.0 - sum((count / n) ** q for count in counts))
    values, counts = np.unique(labels2, return_counts=True)
    H_vq = (1.0 / (q - 1.0)) * (1.0 - sum((count / n) ** q for count in counts))
    # Calculate H_q(U,V)
    contingency = contingency_table(labels1, labels2)
    sum_nij = sum((nij / n) ** q for nij in contingency.flatten())
    H_uvq = (1.0 / (q - 1.0)) * (1.0 - sum_nij)

    return H_uq, H_vq, H_uvq

def entropy_fixed(labels):
    """
    Calculate the entropy H(G) for a specific clustering result.
    Formula: - sum (g_j/N) * log(g_j/N)
    """
    labels = np.array(labels)
    N = len(labels)
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / N
    #Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]

    return -np.sum(probs * np.log(probs))


