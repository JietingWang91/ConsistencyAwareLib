from evaluation_index.utils.entropy import *
from evaluation_index.baseline.MI import MI

def NMI(labels1, labels2,method="sum"):
    """
    Calculates Normalized Mutual Information (NMI).

    Parameters:
    labels1: List of labels for the first clustering (length N).
    labels2: List of labels for the second clustering (length N).
    method : str, default: "sum"
    The normalization method (upper bound) to use.
    Can be any of {"min", "max", "sqrt", "sum", "joint"}:
        - min:   min(H_u, H_v)
        - max:   max(H_u, H_v)
        - sqrt:  np.sqrt(H_u * H_v)
        - sum:   (H_u + H_v) / 2.0
        - joint: H_uv

    The relationship between these upper bounds is:
    MI <= min <= sqrt <= sum <= max <= joint

    Returns:
    nmi: The Normalized Mutual Information, calculated as Mutual Information divided by the selected upper bound.
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    H_u, H_v, H_uv=entropy(labels1, labels2)

    I = MI(labels1,labels2)

    if method == "min":
        denom = min(H_u, H_v)
    elif method == "max":
        denom = max(H_u, H_v)
    elif method == "sqrt":
        denom = np.sqrt(H_u*H_v)
    elif method == "sum":
        denom = (H_u+H_v) / 2.0
    elif method == "joint":
        denom = H_uv
    else:
        raise ValueError(
            "method must be one of {'min', 'max', 'sqrt', 'sum', 'joint'}"
        )
    if np.isclose(denom, 0.0):
        return 1.0 if partitions_equivalent(labels1, labels2) else 0.0
    if np.isclose(I, 0.0):
        return 0.0

    nmi=I / denom
    return nmi