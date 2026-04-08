from evaluation_index.utils.entropy import *

def MIq(labels1, labels2, q):
    """
    Calculate the Generalized Tsallis Mutual Information (MI_q).

    The Tsallis Mutual Information is defined as:
    MI_q(U, V) = H_q(U) + H_q(V) - H_q(U, V)

    where H_q is the Tsallis entropy. Note that unlike Shannon MI,
    Tsallis MI depends on the entropic index q.

    Parameters:
        labels1: List or array of labels for the first clustering.
        labels2: List or array of labels for the second clustering.
        q:       The entropic index (q != 1).

    Returns:
    mi_q: The calculated Tsallis Mutual Information value.
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    H_uq, H_vq, H_uvq=tsallis_q_entropy(labels1, labels2, q)
    #Formula：MIq=H_uq + H_vq - H_uvq
    mi_q=H_uq + H_vq - H_uvq

    return mi_q