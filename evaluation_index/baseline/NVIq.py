from evaluation_index.utils.permutation import *
from evaluation_index.utils.entropy import *
from evaluation_index.baseline.VIq import VIq

def NVIq(labels1, labels2, q):
    """
    Calculate the Normalized Variation of Information (NVI_q) based on Tsallis entropy.

    The metric is normalized by the expected value of VI_q under the Permutation Model.
     NVI_q = VI_q / E[VI_q]
    """
    # E[VI_q] = 2 * E[H_q(U,V)] - H_q(U) - H_q(V)
    E_H_UV=expected_joint_tsallis_entropy(labels1, labels2, q)
    H_U ,H_V ,_ =tsallis_q_entropy(labels1, labels2, q)
    # Note: Marginal entropies H_q(U) and H_q(V) are fixed under the Permutation Model (Fixed Marginals)
    E_VI_q = 2 * E_H_UV - H_U - H_V

    # Calculates NVI_q
    # NVI_q = VI_q / E[VI_q]
    vi_q=VIq(labels1, labels2, q)
    #Prevent division by zero if the denominator is 0
    if np.isclose(E_VI_q, 0):
        return 0.0

    nvi_q = vi_q / E_VI_q

    return nvi_q