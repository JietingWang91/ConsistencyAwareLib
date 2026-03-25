import numpy as np
from sklearn.metrics import accuracy_score


def StandardizedAccuracy(predicted_labels, true_labels, num_classes):
    """
    Calculate Standardized Accuracy (SA) metric which measures how significantly the classifier
    performs better than random chance, standardized by the expected variance.

    Args:
        predicted_labels: List/array of predicted labels, supports discrete values
        true_labels: List/array of ground truth labels , supports discrete values
        num_classes: Integer representing the number of distinct classes

    Returns:
        float: Standardized Accuracy score (real part if complex)
    """
    # Convert all labels to integers for consistent processing
    predicted_labels = [int(label) for label in predicted_labels]
    true_labels = [int(label) for label in true_labels]
    N = len(true_labels)

    # Create mapping from class labels to 0-based indices
    class_to_index = {class_label: idx for idx, class_label in enumerate(range(1, num_classes + 1))}

    # Initialize counters for predicted and true class distributions
    predicted_counts = [0] * num_classes
    true_counts = [0] * num_classes

    # Count class occurrences in predictions and true labels
    for label in predicted_labels:
        predicted_counts[class_to_index[label]] += 1
    for label in true_labels:
        true_counts[class_to_index[label]] += 1

    # Calculate standard accuracy and expected random accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    expected_accuracy = sum(tc * pc for tc, pc in zip(true_counts, predicted_counts)) / (N ** 2)

    # Calculate variance term with Laplace smoothing
    variance_term = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            # Main components of the complex variance calculation
            true_i = true_counts[i]
            pred_j = predicted_counts[j]
            true_j = true_counts[j]
            pred_i = predicted_counts[i]

            numerator = (true_i * pred_j) * (true_j * pred_i) * (N - true_j) * (N - pred_i)
            probability_terms = (1 + (true_j * pred_i - 1) / (N * (N - 1)) + (1 - pred_i - true_j) / N)

            # Complete numerator with Laplace smoothing
            complete_numerator = numerator * probability_terms + 1

            # Denominator with Laplace smoothing
            denominator = (N * N) + num_classes

            variance_term += complete_numerator / denominator

    # Calculate variance and standardized accuracy
    variance = variance_term - expected_accuracy ** 2
    standardized_accuracy = (accuracy - expected_accuracy) / np.sqrt(variance)

    return np.real(standardized_accuracy)