from evaluation_index.utils.entropy import *
from evaluation_index.baseline.MIq import MIq

def NMIq(labels1, labels2, q, method="sum"):
    """
    Calculate the Generalized Normalized Mutual Information (NMI_q).

    This metric normalizes the Tsallis Mutual Information (MI_q) by an upper bound
    determined by the entropies of the individual clusterings.

    Parameters:
    labels1: List or array of labels for the first clustering (length N).
    labels2: List or array of labels for the second clustering (length N).
    q:       The entropic index (q != 1).
    method:  str, default: "sum"
             The normalization method (denominator) to use.
             Options:
             - "min":   min(H_uq, H_vq)
             - "max":   max(H_uq, H_vq)
             - "sqrt":  sqrt(H_uq * H_vq) (Geometric mean)
             - "sum":   (H_uq + H_vq) / 2.0 (Arithmetic mean)
             - "joint": H_uvq (Joint entropy)

    Returns:
    nmiq: The Normalized Tsallis Mutual Information value (0.0 to 1.0).
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    H_uq, H_vq, H_uvq=tsallis_q_entropy(labels1, labels2, q)

    miq = MIq(labels1,labels2,q)

    if miq == 0.0:
        return 0.0

    if method == "min":
        denom = min(H_uq, H_vq)
    elif method == "max":
        denom = max(H_uq, H_vq)
    elif method == "sqrt":
        denom = np.sqrt(H_uq*H_vq)
    elif method == "sum":
        denom = (H_uq+H_vq) / 2.0
    elif method == "joint":
        denom = H_uvq
    else:
        raise ValueError(
            "method must be one of {'min', 'max', 'sqrt', 'sum', 'joint'}"
        )

    nmiq=miq / denom
    return nmiq