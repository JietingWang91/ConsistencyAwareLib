from evaluation_index.utils.entropy import *

def MI(labels1, labels2):
    """
    Calculates Mutual Information
    Parameters:
    labels1: List of labels for the first clustering (length N).
    labels2: List of labels for the second clustering (length N).

    Returns:
    mi: The Mutual Information value.
    """
    # Calculate the number of each pair
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    H_u, H_v, H_uv=entropy(labels1, labels2)
    #Formula：MI=H_u + H_v - H_uv
    mi=H_u + H_v - H_uv

    return mi