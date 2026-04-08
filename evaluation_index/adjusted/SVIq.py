from evaluation_index.utils.permutation import *
from evaluation_index.utils.variance import *
from evaluation_index.baseline.VIq import VIq

def SVIq(labels1, labels2, q):
    """
        Calculates the Standardized Variation of Information (SVI_q).

        Formula: SVI_q = (Sum(n_ij^q) - E[Sum(n_ij^q)]) / sqrt(Var(Sum(n_ij^q)))
        (Note: This formulation converts the VI distance metric into a standardized similarity measure.)

        Note: Computational complexity is O(N^3); it might be slow for datasets with N > 500.

        Parameters:
            labels1: Array of true labels.
            labels2: Array of predicted labels.
            q: Tsallis entropy parameter (q != 1).

        Returns:
        float: SVI_q index value.
    """
    if q == 1:
        raise ValueError("The formula denominator is undefined when q=1. Please use the approximation for Shannon SVI.")

    # E[VI_q] = 2 * E[H_q(U,V)] - H_q(U) - H_q(V)
    E_H_UV = expected_joint_tsallis_entropy(labels1, labels2, q)
    H_U, H_V, _ = tsallis_q_entropy(labels1, labels2, q)
    #Note: Marginal entropies H_q(U) and H_q(V) are fixed under the random permutation model (Fixed Marginals)
    E_VI_q = 2 * E_H_UV - H_U - H_V

    #Calculates VI_q
    vi_q=VIq(labels1, labels2, q)

    var_viq = 4 * var_hq_score(labels1, labels2, q)

    if var_viq <= 0:
        return 0.0
    #Calculates SVI_q
    sviq = (E_VI_q - vi_q) / np.sqrt(var_viq)

    return sviq