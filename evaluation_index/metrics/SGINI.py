import numpy as np
from scipy.special import gammainc

def SGINI(X, Y):
    """
    Calculate the normalized Gini index between two categorical variables X and Y.

    Parameters:
    X, Y : array-like
        Input arrays containing categorical data

    Returns:
    float: Normalized Gini index (SGI_)
    """
    contingency_table, x_marginals, y_marginals = create_contingency_table(X, Y)
    num_x_categories, num_y_categories = contingency_table.shape

    # Handle case when X has only one category
    if num_x_categories == 1:
        return -np.inf

    total_samples = np.sum(contingency_table)
    x_marginals = np.where(x_marginals == 0, 1, x_marginals)  # Smooth zero counts

    # Calculate GiniColumns (Gini index for Y)
    gini_columns = 1 - np.sum((y_marginals / total_samples) ** 2)

    # Calculate raw Gini index
    gini_matrix = (contingency_table ** 2) / np.outer(x_marginals, np.ones(num_y_categories)) / total_samples
    raw_gini = 1 - np.sum(gini_matrix)
    adjusted_gini = gini_columns - raw_gini

    # Calculate expected Gini under independence
    expected_gini = ((num_x_categories - 1) / (total_samples - 1)) * gini_columns

    # Calculate variance components
    gini_variance = calculate_gini_variance(x_marginals, y_marginals, total_samples)
    y_squared_sum = np.sum((y_marginals / total_samples) ** 2)
    gini_v = -y_squared_sum ** 2 - 2 * y_squared_sum * expected_gini + gini_variance
    variance = gini_v - (expected_gini) ** 2

    # Calculate normalized Gini index
    normalized_gini = (adjusted_gini - expected_gini) / np.sqrt(variance)

    return normalized_gini


def create_contingency_table(X, Y):
    """
    Create a contingency table between two categorical variables.

    Parameters:
    X, Y : array-like
        Input arrays containing categorical data

    Returns:
    tuple: (contingency_table, x_marginals, y_marginals)
        contingency_table: 2D array of counts
        x_marginals: Counts for each category in X
        y_marginals: Counts for each category in Y
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    x_categories = np.unique(X)
    y_categories = np.unique(Y)

    contingency_table = np.zeros((len(x_categories), len(y_categories)))
    x_marginals = np.zeros(len(x_categories))
    y_marginals = np.zeros(len(y_categories))

    for i, x_val in enumerate(x_categories):
        x_marginals[i] = np.sum(X == x_val)
        for j, y_val in enumerate(y_categories):
            y_marginals[j] = np.sum(Y == y_val)
            contingency_table[i, j] = np.sum((X == x_val) & (Y == y_val))

    return contingency_table, x_marginals, y_marginals


def calculate_gini_variance(x_marginals, y_marginals, total_samples):
    """
    Calculate the variance components for Gini index calculation.

    Parameters:
    x_marginals : array-like
        Counts for each category in X
    y_marginals : array-like
        Counts for each category in Y
    total_samples : int
        Total number of observations

    Returns:
    float: Combined variance of all components
    """
    smoothing_term = 0  # Originally 'lapalas_n'

    # Initialize term matrices
    y_terms = {
        'y3': np.zeros((len(x_marginals), len(y_marginals))),
        'y2': np.zeros((len(x_marginals), len(y_marginals))),
        'y1': np.zeros((len(x_marginals), len(y_marginals))),
        'y0': np.zeros((len(x_marginals), len(y_marginals)))
    }

    # Calculate y terms for first variance component
    for i, K in enumerate(x_marginals):
        for j, M in enumerate(y_marginals):
            y_terms['y3'][i, j] = (
                    K * (K - 1) * (K - 2) * (K - 3) * M * (M - 1) * (M - 2) * (M - 3) /
                    (total_samples * (total_samples - 1) * (total_samples - 2) * (total_samples - 3)))
            y_terms['y2'][i, j] = (
                    K * (K - 1) * (K - 2) * M * (M - 1) * (M - 2) /
                    (total_samples * (total_samples - 1) * (total_samples - 2)))
            y_terms['y1'][i, j] = (
                    K * (K - 1) * M * (M - 1) /
                    (total_samples * (total_samples - 1)))
            y_terms['y0'][i, j] = K * M / total_samples

    # Calculate four variance components
    variance_components = [0, 0, 0, 0]

    # First component (same x and y categories)
    term1_matrix = (
            (y_terms['y0'] + 7 * y_terms['y1'] + 6 * y_terms['y2'] + y_terms['y3']) /
            ((x_marginals * total_samples) ** 2))
    variance_components[0] = np.sum(term1_matrix)

    # Second component (different x categories)
    term2_matrix = np.zeros((len(x_marginals), len(y_marginals), len(x_marginals)))
    for i, K in enumerate(x_marginals):
        for j, M in enumerate(y_marginals):
            for i_, K_ in enumerate(x_marginals):
                if i_ != i:
                    C1 = (K_ + smoothing_term) / (total_samples - K + smoothing_term)
                    C4 = (K_ * (K_ - 1 + smoothing_term)) / (total_samples - K) / (
                                total_samples - K - 1 + smoothing_term)

                    coeffs = calculate_variance_coefficients(M, C1, C4)
                    term2_matrix[i, j, i_] = calculate_term_value(
                        coeffs, y_terms, i, j, total_samples * K, total_samples * K_)

    variance_components[1] = np.sum(term2_matrix)

    # Third component (different y categories)
    term3_matrix = np.zeros((len(x_marginals), len(y_marginals), len(y_marginals)))
    for i, K in enumerate(x_marginals):
        for j, M in enumerate(y_marginals):
            for j_, M_ in enumerate(y_marginals):
                if j_ != j:
                    C1 = (M_ + smoothing_term) / (total_samples - M + smoothing_term)
                    C4 = ((M_ + smoothing_term) * (M_ - 1 + smoothing_term)) / (
                            total_samples - M + smoothing_term) / (total_samples - M - 1 + smoothing_term)

                    coeffs = calculate_variance_coefficients(K, C1, C4)
                    term3_matrix[i, j, j_] = calculate_term_value(
                        coeffs, y_terms, i, j, total_samples * K, total_samples * K)

    variance_components[2] = np.sum(term3_matrix)

    # Fourth component (different x and y categories)
    term4 = 0
    for i, K in enumerate(x_marginals):
        for j, M in enumerate(y_marginals):
            for i_, K_ in enumerate(x_marginals):
                if i_ != i:
                    for j_, M_ in enumerate(y_marginals):
                        if j_ != j:
                            a = (K_ + smoothing_term) / (total_samples - K + smoothing_term)
                            b = (K_ + smoothing_term) * (K_ - 1 + smoothing_term) / (
                                    total_samples - K - 1 + smoothing_term) / (total_samples - K + smoothing_term)

                            C2 = b * ((M_ + smoothing_term) * (M_ - 1 + smoothing_term)) / (
                                    total_samples - M + smoothing_term) / (total_samples - M - 1 + smoothing_term)
                            C1 = (2 * b * (1 - M_) - a) * (M_ + smoothing_term) / (total_samples - M + smoothing_term)
                            C0 = M_ * (a - b) + b * M_ * M_

                            C4 = C2
                            C3 = C2 - C1 - 2 * C2 * K
                            C2_combined = C0 + (C1 - C2) * K + C2 * K * K

                            coeffs = {
                                'C1': C2_combined + C3 + C4,
                                'C2': 7 * C4 + 3 * C3 + C2_combined,
                                'C3': 6 * C4 + C3,
                                'C4': C4
                            }
                            term4 += calculate_term_value(
                                coeffs, y_terms, i, j, total_samples * K, total_samples * K_)

    variance_components[3] = term4

    return sum(variance_components)


def calculate_variance_coefficients(base, C1, C4):
    """Helper function to calculate variance coefficients."""
    C2 = base * (C1 - C4) + base ** 2 * C4
    C3 = -2 * base * C4 + C4 - C1

    return {
        'C1': C2 + C3 + C4,
        'C2': 7 * C4 + 3 * C3 + C2,
        'C3': 6 * C4 + C3,
        'C4': C4
    }


def calculate_term_value(coefficients, y_terms, i, j, denom1, denom2):
    """Helper function to calculate term value using coefficients."""
    return (
            (1 / denom1 / denom2) *
            (coefficients['C4'] * y_terms['y3'][i, j] +
             coefficients['C3'] * y_terms['y2'][i, j] +
             coefficients['C2'] * y_terms['y1'][i, j] +
             coefficients['C1'] * y_terms['y0'][i, j])
    )