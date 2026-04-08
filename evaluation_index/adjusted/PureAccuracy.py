from sklearn.metrics import accuracy_score

def PureAccuracy(predicted_labels, true_labels, num_classes):
    """
    Calculate the Pure Accuracy (PA) metric by accounting for expected random agreement.

    Args:
        predicted_labels: List/array of predicted labels, supports discrete values
        true_labels: List/array of ground truth labels, supports discrete values
        num_classes: Integer representing the number of distinct classes

    Returns:
        float: Purity Accuracy score
    """

    # Convert all labels to integers for consistent processing
    predicted_labels = [int(label) for label in predicted_labels]
    true_labels = [int(label) for label in true_labels]

    # Create mapping from class labels to 0-based indices
    class_to_index = {class_label: idx for idx, class_label in enumerate(range(1, num_classes + 1))}

    # Initialize counters for predicted and true class distributions
    predicted_class_counts = [0] * num_classes
    true_class_counts = [0] * num_classes

    # Count occurrences of each class in predictions
    for label in predicted_labels:
        predicted_class_counts[class_to_index[label]] += 1

    # Count occurrences of each class in true labels
    for label in true_labels:
        true_class_counts[class_to_index[label]] += 1

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate expected accuracy (EA) under random classification
    expected_accuracy = sum(
        true_count * predicted_count
        for true_count, predicted_count in zip(true_class_counts, predicted_class_counts)
    ) / (len(true_labels) ** 2)

    # Calculate Pure Accuracy (adjusted for random agreement)
    if expected_accuracy == 1:  # Prevent division by zero
        return 0.0
    pure_accuracy = (accuracy - expected_accuracy) / (1 - expected_accuracy)

    return pure_accuracy