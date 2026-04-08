from evaluation_index.utils.entropy import *
from scipy.stats import hypergeom


def phi_shannon(n):
    """
    Core function for Shannon entropy term: n * ln(n).
    Definitions: 0 * ln(0) = 0.
    """
    if n <= 0:
        return 0.0
    return n * np.log(n)


def expected_phi_value(M, n, N):
    """
    Calculate E[n * ln(n)], where n ~ Hyp(M, n_success, N_sample).
    (Expectation of the Shannon term under Hypergeometric distribution).

    Parameters:
    M: Population size
    n: Number of successes in population
    N: Sample size
    """
    if N <= 0 or n <= 0 or M <= 0:
        return 0.0

    k_min = max(0, n + N - M)
    k_max = min(n, N)

    if k_min > k_max:
        return 0.0

    k_values = np.arange(k_min, k_max + 1)
    probs = hypergeom.pmf(k_values, M, n, N)

    # Calculate sum(P(k) * k * ln(k))
    # Avoid log(0) by masking
    phi_vals = np.zeros_like(k_values, dtype=np.float64)
    mask = k_values > 0
    phi_vals[mask] = k_values[mask] * np.log(k_values[mask])

    return np.sum(phi_vals * probs)

def expected_nq(M, n, N, q):
    """
    Helper function: Calculate E[X^q] for hypergeometric distribution X ~ Hyp(M, n, N).

    Parameters:
    M (int): Population size
    n (int): Number of successes in population
    N (int): Number of draws (Sample size)
    q (float): Exponent q

    Returns:
    float: E[X^q]
    """
    # Boundary checks
    if N <= 0 or n <= 0 or M <= 0:
        return 0.0

    #Determine range of X (k): [max(0, n+N-M), min(n, N)]
    k_min = max(0, n + N - M)
    k_max = min(n, N)

    if k_min > k_max:
        return 0.0

    k_values = np.arange(k_min, k_max + 1)

    # Calculate Probability Mass Function P(X=k)
    probs = hypergeom.pmf(k_values, M, n, N)

    #Calculate E[X^q] = sum(k^q * P(k))
    # Handle 0^q (which is 0 for q>0)
    k_q = np.power(k_values.astype(np.float64), q)

    return np.sum(k_q * probs)


def var_hq_score(labels1, labels2, q):
    """
    Calculate the variance Var(H_q) of the Generalized Tsallis Entropy H_q(U, V).

    This function computes the exact variance under the Permutation Model using
    second-order moments of the hypergeometric distribution.

    Parameters:
    labels1 (array-like): True clustering labels
    labels2 (array-like): Predicted clustering labels
    q (float): Tsallis entropy parameter (q != 1)

    Returns:
    float: Var(H_q(U, V))
    """
    if q == 1:
        raise ValueError("q=1 results in division by zero. Use the dedicated Shannon entropy formula or let q approach 1.")

    # 1. Construct Contingency Table and Marginal Distributions
    n_ij_mat = contingency_table(labels1, labels2)
    R, C = n_ij_mat.shape

    a = np.sum(n_ij_mat, axis=1)  #Row sums (a_i)
    b = np.sum(n_ij_mat, axis=0)  #Column sums (b_j)
    N_total = np.sum(n_ij_mat)  # Total samples N

    # 2. Calculate First Moment: Sum(E[n_ij^q])
    sum_expected_nq = 0.0

    for i in range(R):
        for j in range(C):
            # n_ij ~ Hyp(N, a_i, b_j)
            sum_expected_nq += expected_nq(N_total, a[i], b[j], q)

    # 3. Calculate Second Moment: E[(Sum n_ij^q)^2]
    # We expand the square of the sum and calculate expectations using conditional independence properties.
    expected_sq_sum = 0.0

    for i in range(R):
        for j in range(C):
            a_i = a[i]
            b_j = b[j]

            # Distribution of n_ij ~ Hyp(N, a_i, b_j)
            k_min = max(0, a_i + b_j - N_total)
            k_max = min(a_i, b_j)
            k_values = np.arange(k_min, k_max + 1)
            probs_k = hypergeom.pmf(k_values, N_total, a_i, b_j)

            #Sum over all possible values k for n_ij
            for idx, k in enumerate(k_values):
                p_nij_k = probs_k[idx]
                k_q = float(k) ** q

                # Term 1: n_ij^q
                term1 = k_q

                # Term 2: Sum_{i' != i} E[n~_i'j^q]
                # n~_i'j ~ Hyp(N-a_i, a_i', b_j-k)
                term2 = 0.0
                if N_total - a_i > 0:  # Avoid empty population
                    for i_prime in range(R):
                        if i_prime == i: continue
                        # Params: M=N-a_i, n=a_i', N=b_j-k
                        term2 += expected_nq(N_total - a_i, a[i_prime], b_j - k, q)

                # Term 3: Sum_{j' != j} E[...]
                term3 = 0.0
                for j_prime in range(C):
                    if j_prime == j: continue

                    # Outer Expectation: Expectation over n~_ij' (let's call it t)
                    # This represents the count in the same row i but different column j', given n_ij=k.
                    # n~_ij' ~ Hyp(N-b_j, b_j', a_i-k)
                    M_t = N_total - b_j
                    n_t = b[j_prime]
                    N_t = a_i - k

                    if M_t <= 0 or N_t < 0:
                        continue

                    t_min = max(0, n_t + N_t - M_t)
                    t_max = min(n_t, N_t)
                    t_values = np.arange(t_min, t_max + 1)
                    probs_t = hypergeom.pmf(t_values, M_t, n_t, N_t)

                    term3_inner_sum = 0.0
                    for idx_t, t in enumerate(t_values):
                        p_t = probs_t[idx_t]
                        t_q = float(t) ** q

                        # Term 3.1: n~_ij'^q
                        sub_term1 = t_q

                        # Term 3.2: Sum_{i' != i} E[n~~_i'j'^q]
                        # n~~_i'j' ~ Hyp(N-a_i, a_i', b_j'-t)
                        sub_term2 = 0.0
                        if N_total - a_i > 0:
                            for i_prime in range(R):
                                if i_prime == i: continue
                                # Params: M=N-a_i, n=a_i', N=b_j'-t
                                sub_term2 += expected_nq(N_total - a_i, a[i_prime], b[j_prime] - t, q)

                        term3_inner_sum += p_t * (sub_term1 + sub_term2)

                    term3 += term3_inner_sum

                # Accumulate into total expected squared sum
                # E[(Sum)^2] += n_ij^q * P(n_ij) * [Term1 + Term2 + Term3]
                # phi(n_ij) = n_ij^q = k_q
                expected_sq_sum += k_q * p_nij_k * (term1 + term2 + term3)

    # 4. Calculate Final Variance
    # Var = (1 / ((q-1)^2 * N^(2q))) * (E[S^2] - (E[S])^2)
    numerator = expected_sq_sum - (sum_expected_nq ** 2)
    denominator = ((q - 1) ** 2) * (float(N_total) ** (2 * q))

    variance = numerator / denominator

    return variance


def variance_shannon_mi(labels1, labels2):
    """
    Calculate the variance Var(MI) of the Shannon Mutual Information.

    Parameters:
    labels1 (array-like): First clustering labels
    labels2 (array-like): Second clustering labels

    Returns:
    float: Var(MI
    """
    # 1. Preparation
    n_ij_mat = contingency_table(labels1, labels2)
    R, C = n_ij_mat.shape
    N = np.sum(n_ij_mat)

    # Marginal sum
    a = np.sum(n_ij_mat, axis=1)  # Row marginals (Cluster sizes in labels1)
    b = np.sum(n_ij_mat, axis=0)  #Column marginals (Cluster sizes in labels2)

    # 2. Calculate First Moment E[S]
    # S = sum(n_ij * ln(n_ij))
    # E[S] = sum_{ij} E[phi(n_ij)]
    expected_sum_phi = 0.0
    for i in range(R):
        for j in range(C):
            # n_ij ~ Hyp(N, a[i], b[j])
            expected_sum_phi += expected_phi_value(N, a[i], b[j])

    # 3. Calculate Second Moment E[S^2]
    # E[S^2] = E[(sum n_ij * ln(n_ij))^2]
    expected_sq_sum = 0.0

    for i in range(R):
        for j in range(C):

            a_i, b_j = a[i], b[j]
            # Iterate possible values k for n_ij
            k_min = max(0, a_i + b_j - N)
            k_max = min(a_i, b_j)
            k_vals = np.arange(k_min, k_max + 1)
            probs_k = hypergeom.pmf(k_vals, N, a_i, b_j)

            for idx_k, k in enumerate(k_vals):
                p_k = probs_k[idx_k]
                val_k = phi_shannon(k)

                # Part 1: phi(n_ij)
                term1 = val_k

                # Part 2: Row interactions: i' != i
                # n~_i'j ~ Hyp(N-a_i, a_i', b_j-k)
                term2 = 0.0
                if N - a_i > 0:
                    for i_p in range(R):
                        if i_p == i: continue
                        term2 += expected_phi_value(N - a_i, a[i_p], b_j - k)

                # Part 3: Column interactions: j' != j
                term3 = 0.0
                for j_p in range(C):
                    if j_p == j: continue

                    # Expectation over n~_ij' (variable t)
                    # n~_ij' ~ Hyp(N-b_j, b_j', a_i-k)
                    M_t = N - b_j
                    n_t = b[j_p]
                    N_t = a_i - k

                    if M_t <= 0 or N_t < 0: continue

                    t_min = max(0, n_t + N_t - M_t)
                    t_max = min(n_t, N_t)
                    t_vals = np.arange(t_min, t_max + 1)
                    probs_t = hypergeom.pmf(t_vals, M_t, n_t, N_t)

                    inner_term = 0.0
                    for idx_t, t in enumerate(t_vals):
                        p_t = probs_t[idx_t]
                        val_t = phi_shannon(t)

                        # Part 3.1: phi(n~_ij')
                        sub1 = val_t

                        # Part 3.2: Cross interactions: i' != i
                        # n~~_i'j' ~ Hyp(N-a_i, a_i', b_j'-t)
                        sub2 = 0.0
                        if N - a_i > 0:
                            for i_p in range(R):
                                if i_p == i: continue
                                sub2 += expected_phi_value(N - a_i, a[i_p], b[j_p] - t)

                        inner_term += p_t * (sub1 + sub2)

                    term3 += inner_term

                # Accumulate E[S^2]
                # Sum_{ij} Sum_{k} phi(k) * P(k) * [Term1 + Term2 + Term3]
                expected_sq_sum += val_k * p_k * (term1 + term2 + term3)

    # 4. Calculate Variance
    # Var(S) = E[S^2] - (E[S])^2
    var_s = expected_sq_sum - (expected_sum_phi ** 2)

    # 5. Return Var(MI)
    # MI = (1/N) * S + Constant
    # Var(MI) = (1/N^2) * Var(S)

    # Handle potential small negative values due to floating point precision
    if var_s < 0:
        var_s = 0.0

    var_mi = var_s / (N ** 2)

    return var_mi