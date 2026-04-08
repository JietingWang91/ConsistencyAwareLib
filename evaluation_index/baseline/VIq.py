from evaluation_index.utils.entropy import *

def VIq(labels1, labels2, q):
    """
    Calculate the Generalized Variation of Information (VI_q) based on Tsallis Entropy.

    Standard VI is defined as VI(U,V) = H(U) + H(V) - 2I(U,V).
    In terms of joint entropy, this is equivalent to VI(U,V) = 2H(U,V) - H(U) - H(V).

    This function applies the same relationship using Generalized Tsallis entropy.

    Parameters:
    labels1: List or array of labels for the first clustering.
    labels2: List or array of labels for the second clustering.
    q:       The entropic index (q != 1).

    Returns:
    vi_q: The Generalized Variation of Information value.
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    H_uq, H_vq, H_uvq = tsallis_q_entropy(labels1, labels2, q)
    # Formula：VIq=2H_uvq - H_uq - H_vq
    vi_q = 2*H_uvq - H_uq - H_vq

    return vi_q