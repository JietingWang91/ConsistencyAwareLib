from evaluation_index.utils.permutation import *
from evaluation_index.baseline.MI import MI
from evaluation_index.utils.contingency import partitions_equivalent, validate_label_inputs
import numpy as np
import math

def AMI(labels1, labels2,model="perm",method="sum",sided='two-sided'):
    """
    Calculates the Adjusted Mutual Information (AMI) under specific models.

    Parameters:
    labels1: List or array
        Labels of the first clustering, length N.
    labels2: List or array
        Labels of the second clustering, length N.

    model : str, default "perm"
        The null model to use. Options: {'perm', 'num', 'all'}.

    method : str, default "sum"
        Normalization method. Can be any of {"min", "max", "sqrt", "sum"}.
        Represents the value used for the upper bound denominator:
        - min: min(H_u, H_v)
        - max: max(H_u, H_v)
        - sqrt: np.sqrt(H_u * H_v)
        - sum: (H_u + H_v) / 2.0

        Note: joint is H_uv.
        The relationship between upper bounds is:
        MI <= min <= sqrt <= sum <= max <= joint

    sided: str, optional ['two-sided', 'one-sided'] (default 'two-sided')
        - 'two-sided': Assumes both A and B are randomly drawn from the model.
        - 'one-sided': Assumes a randomly generated clustering A (with fixed cluster count K_A)
          against a fixed reference clustering B.

    Returns:
    ami: float
        The Adjusted Mutual Information value.
    """
    # Calculate contingency table and marginal sums
    labels1, labels2 = validate_label_inputs(labels1, labels2)
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, _ = marginal_sums(n_ij)
    N = labels1.shape[0]
    K_A = len(a_i)
    K_B = len(b_j)
    H_u, H_v, _ = entropy(labels1, labels2)

    if model == "perm":
        if sided == 'two-sided' or sided == 'one-sided':
            e_mi=expected_mi_perm(labels1, labels2)
        else:
            raise ValueError(f"Unknown sided: {sided}. Choose from ['two-sided', 'one-sided']")
        if method == "min":
            denom = min(H_u, H_v)
        elif method == "max":
            denom = max(H_u, H_v)
        elif method == "sqrt":
            denom = np.sqrt(H_u * H_v)
        elif method == "sum":
            denom = (H_u + H_v) / 2.0
        else:
            raise ValueError(
                "method must be one of {'min', 'max', 'sqrt', 'sum'}"
            )
    elif model == "num":
        if sided == 'two-sided':
            e_mi = expected_mi_num_twosided(labels1, labels2)
        elif sided == 'one-sided':
            e_mi = expected_mi_num_onesided(labels1, labels2)
        if method == "min":
            denom = min(math.log(K_A), math.log(K_B))
        elif method == "max":
            denom = max(math.log(K_A), math.log(K_B))
        elif method == "sqrt":
            denom = np.sqrt(math.log(K_A) * math.log(K_B))
        elif method == "sum":
            denom = (math.log(K_A) + math.log(K_B)) / 2.0
        else:
            raise ValueError(
                "method must be one of {'min', 'max', 'sqrt', 'sum'}"
            )
    elif model == "all":
        if sided == 'two-sided':
            e_mi = expected_mi_all_twosided(labels1, labels2)
        elif sided == 'one-sided':
            e_mi = expected_mi_all_onesided(labels1, labels2)
        if method == "min" or method == "max" or method == "sqrt" or method == "sum":
            denom = math.log(N)
        else:
            raise ValueError(
                "method must be one of {'min', 'max', 'sqrt', 'sum'}"
            )
    else:
        raise ValueError(
            "model must be one of {'perm', 'num', 'all'}"
        )

    I = MI(labels1, labels2)
    denominator = denom - e_mi
    if np.isclose(denominator, 0.0):
        return 1.0 if partitions_equivalent(labels1, labels2) else 0.0

    numerator = I - e_mi
    ami = numerator / denominator
    return ami