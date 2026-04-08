from evaluation_index.utils.variance import *
from evaluation_index.utils.entropy import *

def SG(labels1, labels2):
    """
    Calculates SG (Standardized G-statistic).

    Basis:
    1. SG = SMI (Standardized Mutual Information)
    2. SG = (MI - E[MI]) / sqrt(Var(MI))
    Note: Computational complexity is O(N^3); suitable for small datasets or cases with few clusters.

    Parameters:
    labels1: True labels (Ground Truth)
    labels2: Predicted labels

    Returns:
    float: SG index value
    """
    #Basic Data Construction
    n_ij = contingency_table(labels1, labels2)
    R, C = n_ij.shape
    N = np.sum(n_ij)
    a = np.sum(n_ij, axis=1)  #Row marginals
    b = np.sum(n_ij, axis=0)  #Column marginals

    #Calculate Observed MI (Shannon)
    # MI = sum(p_ij * ln(p_ij / (p_i * p_j)))
    #    = (1/N) * (sum(nij * ln(nij)) - sum(ai * ln(ai)) - sum(bj * ln(bj))) + ln(N)

    term_nij = np.sum([phi_shannon(x) for x in n_ij.flatten()])
    term_a = np.sum([phi_shannon(x) for x in a])
    term_b = np.sum([phi_shannon(x) for x in b])

    mi_obs = (term_nij - term_a - term_b) / N + np.log(N)

    #Calculate Expected MI: E[MI]
    #Since ai and bj are fixed, E[MI] depends only on E[sum(nij * ln(nij))]
    # E[sum(nij * ln(nij))] = sum_ij E[phi(nij)]

    expected_sum_phi = 0.0
    for i in range(R):
        for j in range(C):
            # nij ~ Hyp(N, a[i], b[j])
            expected_sum_phi += expected_phi_value(N, a[i], b[j])

    e_mi = (expected_sum_phi - term_a - term_b) / N + np.log(N)

    #Calculate Variance Var(MI)
    # Var(MI) = (1/N^2) * Var(sum(nij * ln(nij)))
    # Var(S) = E[S^2] - (E[S])^2

    var_mi = variance_shannon_mi(labels1, labels2)

    #Calculate SG
    if var_mi <= 1e-10:
        return 0.0

    sg = (mi_obs - e_mi) / np.sqrt(var_mi)

    return sg