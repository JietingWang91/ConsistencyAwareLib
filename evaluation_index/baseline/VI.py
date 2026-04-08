from evaluation_index.utils.entropy import *


def VI(labels1, labels2):
    """
    Calculates Variation of Information (VI).

    Parameters:
    labels1: List or array of labels for the first clustering (length N).
    labels2: List or array of labels for the second clustering (length N).

    Returns:
    vi: The Variation of Information value.
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    H_u, H_v, H_uv=entropy(labels1, labels2)
    #Formula：MI=H_u + H_v - H_uv
    Vi=2 * H_uv - H_u -H_v

    return Vi