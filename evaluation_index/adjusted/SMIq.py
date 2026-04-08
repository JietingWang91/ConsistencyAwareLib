from evaluation_index.utils.permutation import *
from evaluation_index.utils.variance import *
from evaluation_index.baseline.MIq import MIq

def SMIq(labels1, labels2, q):
    """
        Calculates Standardized Mutual Information (SMI_q).

        Formula: SMI_q = (MI_q - E[MI_q]) / sqrt(Var(MI_q))

        Parameters:
            labels1: True labels
            labels2: Predicted labels
            q: Tsallis entropy parameter (q != 1)
    """
    if q == 1:
        raise ValueError("Denominator is zero when q=1. Please use the approximation for Shannon SMI.")
    E_H_UV=expected_joint_tsallis_entropy(labels1, labels2, q)
    H_U ,H_V ,_ =tsallis_q_entropy(labels1, labels2, q)
    #Note: Marginal entropies H_q(U) and H_q(V) are fixed under the random permutation model (Fixed Marginals)
    E_MI_q = H_U + H_V - E_H_UV

    #Calculate MI_q
    mi_q=MIq(labels1, labels2, q)

    var_miq = var_hq_score(labels1, labels2, q)

    if var_miq <= 0:
        return 0.0
    #Calculate SMI_q
    smiq = (mi_q - E_MI_q) / np.sqrt(var_miq)

    return smiq