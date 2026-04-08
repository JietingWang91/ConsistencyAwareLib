from evaluation_index.utils.permutation import *
from evaluation_index.utils.entropy import *
from evaluation_index.baseline.MIq import MIq
from evaluation_index.utils.contingency import partitions_equivalent
import numpy as np

def AMIq(labels1, labels2, q,method="sum"):
    """
    Calculates Adjusted Mutual Information based on Tsallis entropy (AMI_q).

    Parameters:
    labels1 (array-like): True clustering labels (Ground truth).
    labels2 (array-like): Predicted clustering labels.
    q : Parameter q for Tsallis entropy (q > 0 and q != 1).
        When q -> 1, it theoretically converges to standard AMI, but this formula handles cases where q != 1.
        When q = 2, it is equivalent to ARI.
    method : str, default "sum"
        Normalization method for the denominator.

    Returns:
    float: AMI_q score.
    """
    if q == 1:
        raise ValueError("This formula applies to q != 1. For q=1, please use standard AMI.")
    E_H_UV=expected_joint_tsallis_entropy(labels1, labels2, q)
    H_U ,H_V ,_ =tsallis_q_entropy(labels1, labels2, q)
    # Note: Marginal entropies H_q(U) and H_q(V) are fixed under the random permutation model (Fixed Marginals)
    E_MI_q = H_U + H_V - E_H_UV

    # Calculate MI_q
    mi_q=MIq(labels1, labels2, q)

    if method == "min":
        denom = min(H_U, H_V)
    elif method == "max":
        denom = max(H_U, H_V)
    elif method == "sqrt":
        denom = np.sqrt(H_U*H_V)
    elif method == "sum":
        denom = (H_U+H_V) / 2.0
    else:
        raise ValueError(
            "method must be one of {'min', 'max', 'sqrt', 'sum'}"
        )

    denominator = denom - E_MI_q
    if np.isclose(denominator, 0.0):
        return 1.0 if partitions_equivalent(labels1, labels2) else 0.0
    # Calculate AMI_q
    amiq = (mi_q - E_MI_q) / denominator

    return amiq