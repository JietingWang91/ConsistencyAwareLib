import torch
import torch.nn.functional as F
import math


def PSED(predicted_features, true_labels):
    """
    Computes a cluster consistency loss that measures how well the predicted features
    form compact clusters matching the true labels.

    Args:
        predicted_features: Tensor of shape (n_samples, n_features) - learned feature vectors
        true_labels: Tensor of shape (n_samples,) - ground truth cluster labels, supports discrete values

    Returns:
        Tensor: The computed consistency loss value
    """
    # Normalize the predicted features
    normalized_features = F.normalize(predicted_features, p=2, dim=1)

    # Compute similarity matrix (cosine similarity between all feature pairs)
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())

    # Calculate counts for each unique label
    unique_labels = torch.unique(true_labels)
    label_counts = [torch.sum(true_labels == label) for label in unique_labels]
    n_samples = len(true_labels)

    # Compute various components of the loss
    total_similarity = torch.sum(similarity_matrix)
    diagonal_similarity = torch.sum(torch.diagonal(similarity_matrix))
    off_diagonal_similarity = total_similarity - diagonal_similarity

    # Compute within-cluster similarity (sum of similarities within each true cluster)
    within_cluster_similarity = 0
    row_start = 0
    for count in label_counts:
        if count > 1:  # Need at least 2 elements to form a submatrix
            row_end = row_start + count
            cluster_submatrix = similarity_matrix[row_start:row_end, row_start:row_end]
            within_cluster_similarity += torch.sum(cluster_submatrix)
        row_start += count

    # Compute weighting coefficient for off-diagonal terms
    off_diagonal_weight = 0
    if n_samples >= 2:
        total_pairs = math.comb(n_samples, 2)
        for count in label_counts:
            if count >= 2:
                cluster_pairs = math.comb(count, 2)
                off_diagonal_weight += cluster_pairs / total_pairs

    # Final loss computation
    consistency_loss = (diagonal_similarity +
                        off_diagonal_weight * off_diagonal_similarity -
                        within_cluster_similarity)

    return consistency_loss