#Random Models and Expectation Correction
from evaluation_index.utils.entropy import *
from scipy.stats import hypergeom
from collections import Counter
import math

def expected_joint_tsallis_entropy(labels1, labels2, q):
    """
    Calculate the Expected Joint Tsallis q-Entropy E[H_q(U,V)] under the
    Permutation Model.

    Formula： E[H_q(U,V)] = 1/(q-1) * (1 - (1/N^q) * sum(E[n_ij^q]))

    Parameters:
    labels1: True label array (U)
    labels2: Predicted label array (V)
    q: Tsallis entropy parameter (q != 1)

    Returns:
    E_H_UV: The expected joint entropy value.
    """
    if np.isclose(q, 1.0):
        raise ValueError("q cannot be 1.0 for Tsallis entropy (division by zero).")

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    N = labels1.shape[0]

    # 1. Get Marginal Distributions (Row and Column Marginals)
    # n_ij is the contingency table; a and b are row/col sums

    n_ij = contingency_table(labels1, labels2)
    a ,b ,_ = marginal_sums(n_ij)

    sum_expected_nij_q = 0.0

    # 2. Iterate through all marginal combinations to compute sum(E[n_ij^q])
    #Under the random model, each (a_i, b_j) pair corresponds to a random variable
    #n_ij following a Hypergeometric distribution.
    for ai in a:
        for bj in b:
            # Determine support range for Hypergeometric dist: k in [max(0, ai+bj-N), min(ai, bj)]
            min_k = max(0, ai + bj - N)
            max_k = min(ai, bj)

            # Generate all possible values for k
            k_values = np.arange(min_k, max_k + 1)

            # Calculate Hypergeometric probability P(n_ij = k)
            # cipy parameters: M=N (Population), n=bj (Successes in Pop), N=ai (Sample Size)
            # Note: Parameter mapping requires care as scipy notation differs from some papers.
            probs = hypergeom.pmf(k_values, N, bj, ai)

            # Calculate term expectation E[n_ij^q] = sum(k^q * P(k))
            term = np.sum((k_values ** q) * probs)

            sum_expected_nij_q += term

    # 3. Calculate final expected entropy
    # Formula: 1/(q-1) * (1 - (1/N^q) * sum(E[n_ij^q]))
    E_H_UV = (1.0 / (q - 1.0)) * (1.0 - (1.0 / (N ** q)) * sum_expected_nij_q)

    return E_H_UV

def stirling_second(n, k):
    """
    Calculate Stirling numbers of the second kind S(n, k).
    Formula: (1/k!) * sum_{j=0}^k (-1)^(k-j) * C(k, j) * j^n
    Note: For very large N, approximate formulas or high-precision libraries
    should be used to avoid overflow.
    """
    if k < 0 or k > n:
        return 0
    sum_val = 0
    for j in range(k + 1):
        term = ((-1) ** (k - j)) * math.comb(k, j) * (j ** n)
        sum_val += term
    return sum_val // math.factorial(k)

def bell_numbers_list(n):
    """
    Generate a list of Bell numbers B_0 to B_n.
    (Rewritten from original to return a list, as required by downstream functions).

    Formula: B_{n+1} = sum_{k=0}^n C(n, k) B_k
    """
    bell = [0] * (n + 1)
    bell[0] = 1
    for i in range(1, n + 1):
        # Calculate B_i using previous Bell numbers
        val = 0
        for k in range(i):
            val += math.comb(i - 1, k) * bell[k]
        bell[i] = val
    return bell

def bell_number(n):
    """
    Calculate single Bell number B_n.
    """
    return bell_numbers_list(n)[-1]

#M_perm E[RI]
#two-sided=one-sided
def expected_ri_perm(labels1, labels2):
    """
    Calculate Expected Rand Index (RI) under the Permutation Model (M_perm).
    Uses the contingency table to calculate marginal distributions.
    """
    # 1. Calculate contingency table n_ij
    n_ij = contingency_table(labels1, labels2)

    # 2. Calculate marginal sums a_i, b_j and total N
    a_i, b_j, N = marginal_sums(n_ij)

    # 3. Calculate sum(C(a_i, 2)) and sum(C(b_j, 2))
    sum_comb_a = sum(math.comb(n, 2) for n in a_i)
    sum_comb_b = sum(math.comb(n, 2) for n in b_j)

    total_pairs = math.comb(N, 2)

    # 4. Calculate probabilities for each side
    p_a = sum_comb_a / total_pairs
    p_b = sum_comb_b / total_pairs

    # 5. E[RI]
    # E[RI] = p_a * p_b + (1 - p_a) * (1 - p_b)
    expected_ri = (p_a * p_b) + ((1 - p_a) * (1 - p_b))

    return expected_ri

#M_num ERI
#two-sided
def expected_ri_num_twosided(labels1, labels2):
    """
    Calculate Expected RI under Fixed Number of Clusters Model (M_num).
    Assumes clusterings are drawn uniformly from the set of all clusterings
    with fixed cluster number K.
    """
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, N = marginal_sums(n_ij)

    # Get number of clusters K_A and K_B
    K_A = len(a_i)
    K_B = len(b_j)

    # Calculate Stirling term: S(N-1, K) / S(N, K)
    # This represents the probability that any two elements belong to the same cluster.
    def prob_same_cluster(n, k):
        denom = stirling_second(n, k)
        if denom == 0: return 0
        return stirling_second(n - 1, k) / denom

    term_a = prob_same_cluster(N, K_A)
    term_b = prob_same_cluster(N, K_B)

    # E[RI] = term_a * term_b + (1 - term_a) * (1 - term_b)
    expected_ri = (term_a * term_b) + ((1 - term_a) * (1 - term_b))

    return expected_ri

#M_num ERI
#one-sided
def expected_ri_num_onesided(labels1, labels2):
    """
    Calculate One-sided Expected RI under M_num.

    Parameters:
    labels1: Random clustering A (Drawn from M_num)
    labels2: Reference clustering G (Fixed structure)

    Formula
    E_num^1[RI(A, G)] = P_A * P_G + (1 - P_A) * (1 - P_G)

    return:expected_ri
    """
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, N = marginal_sums(n_ij)

    #Calculate P_A: Prob that 2 points are in same cluster in Random A
    K_A = len(a_i)

    denom_stirling = stirling_second(N, K_A)
    if denom_stirling == 0:
        p_a = 0.0
    else:
        p_a = stirling_second(N - 1, K_A) / denom_stirling

    #Calculate P_G: Prob that 2 points are in same cluster in Reference G
    sum_comb_g = sum(math.comb(g, 2) for g in b_j)
    total_pairs = math.comb(N, 2)

    if total_pairs == 0:
        p_g = 0.0
    else:
        p_g = sum_comb_g / total_pairs

    expected_ri = (p_a * p_g) + ((1 - p_a) * (1 - p_g))

    return expected_ri

#M_all ERI
#two-side
def expected_ri_all_twosided(labels1, labels2):
    """
    Calculate Expected RI under All Clusterings Model (M_all).
    Assumes clusterings are drawn uniformly from the set of ALL possible clusterings.
    """
    n_ij = contingency_table(labels1, labels2)
    _, _, N = marginal_sums(n_ij)

    bell_nums = bell_numbers_list(N)
    bn = bell_nums[N]
    if bn == 0: return 0

    bn_minus_1 = bell_number(N - 1)

    term = bn_minus_1 / bn

    # E[RI] = term^2 + (1 - term)^2
    expected_ri = (term ** 2) + ((1 - term) ** 2)

    return expected_ri

#M_all ERI
#one-sided
def expected_ri_all_onesided(labels1, labels2):
    """
    Calculate One-sided Expected RI under M_all.

    E_all^1[RI(A, G)] = P_A * P_G + (1 - P_A) * (1 - P_G)

    P_A = B_{N-1} / B_N
    P_G = sum(C(g_j, 2)) / C(N, 2)

    """
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, N = marginal_sums(n_ij)

    #Calculate P_A: Prob that 2 points are in same cluster in Random A
    # P_A = B_{N-1} / B_N
    bell_nums = bell_numbers_list(N)

    #The probability depends on the Random Model (M_all) over N points
    bn = bell_nums[N]
    bn_minus_1 = bell_nums[N - 1]

    if bn == 0:
        p_a = 0.0
    else:
        p_a = bn_minus_1 / bn
    if bn == 0:
        p_a = 0.0
    else:
        p_a = bn_minus_1 / bn

    # Calculate P_G: Prob that 2 points are in same cluster in Fixed G
    # P_G = Q_1^G / C(N, 2) = sum(binom(g_j, 2)) / binom(N, 2)
    sum_comb_g = sum(math.comb(g, 2) for g in b_j)
    total_pairs = math.comb(N, 2)

    if total_pairs == 0:
        p_g = 0.0
    else:
        p_g = sum_comb_g / total_pairs

    # E = p_a * p_g + (1 - p_a) * (1 - p_g)
    expected_ri = (p_a * p_g) + ((1 - p_a) * (1 - p_g))

    return expected_ri

#Calculate E_perm[H(A, B)] under the perm Model
#two-sided=one-sided
def expected_joint_entropy_perm(labels1, labels2):
    """
    Calculate Expected Joint Entropy E_perm[H(A, B)] under the Permutation Model.

    Formula: E[H(A,B)] = - sum_i sum_j sum_n (n/N) * log(n/N) * P(n|a_i, b_j, N)
    Where P(n) follows Hypergeometric distribution Hyp(n|N, a_i, b_j)

    Parameters:
    labels1: True labels (A)
    labels2: Predicted labels (B)
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    N = labels1.shape[0]

    # 1. Get marginals a_i and b_j
    n_ij = contingency_table(labels1, labels2)
    a, b, _ = marginal_sums(n_ij)

    e_joint_entropy = 0.0

    # 2. Iterate all cluster size combinations (a_i, b_j)
    for ai in a:
        for bj in b:
            # 3. Determine range of overlap n: [max(1, ai+bj-N), min(ai, bj)]
            #If n=0, (n/N)*log(n/N) is 0, skip to avoid math domain errors
            start_n = max(1, ai + bj - N)
            end_n = min(ai, bj)

            if start_n > end_n:
                continue

            n_values = np.arange(start_n, end_n + 1)

            # 4. Calculate Hypergeometric probabilities P(n)

            probs = hypergeom.pmf(n_values, N, ai, bj)

            # 5. Calculate joint entropy term: - (n/N) * log(n/N)
            # Note: Formula has a negative sign. We calc positive here and subtract.
            term = (n_values / N) * np.log(n_values / N)

            # Weighted sum for expectation
            e_joint_entropy -= np.sum(term * probs)

    return e_joint_entropy

#Calculate E[MI] under the M_perm model
#two-sided=one-sided
def expected_mi_perm(labels1, labels2):
    """
    Calculate Expected Mutual Information E[MI] under the Permutation Model.

    E_perm[MI] = E[H_A] + E[H_B] - E[H_AB]

    Parameters:
    labels1: True labels (A)
    labels2: Predicted labels (B)

    return:
    expected_mi: Expected mutual information value
    """
    H_u, H_v, _ = entropy(labels1, labels2)
    e_h_uv = expected_joint_entropy_perm(labels1, labels2)
    expected_mi=H_u + H_v - e_h_uv

    return expected_mi

#M_num E_num[H(A)]
#two-sided=one-sided
def expected_entropy_num(labels):
    """
    Calculate Expected Entropy E_num[H(A)] under M_num.
    """
    labels = np.array(labels)
    N = len(labels)
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    if K == 1:
        return 0.0

    denom = stirling_second(N, K)
    if denom == 0: return 0.0

    e_entropy = 0.0

    # Iterate size a: range 1 to N-(K-1)
    for a in range(1, N - (K - 1) + 1):
        # 1. Calculate probability weight P(size=a)
        num_ways = comb(N, a) * stirling_second(N - a, K - 1)
        prob_a = num_ways / denom

        # 2. Entropy term (a/N) * log(a/N)
        if a > 0:
            term_ent = (a / N) * np.log(a / N)
        else:
            term_ent = 0

        e_entropy += prob_a * term_ent

    return -e_entropy

#M_num E_num[H(A, B)]
#two-sided
def expected_joint_entropy_num(labels1, labels2):
    """
    Calculate Expected Joint Entropy E_num[H(A, B)] under M_num.
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    N = len(labels1)
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, _ = marginal_sums(n_ij)

    K_A = len(a_i)
    K_B = len(b_j)

    denom_A = stirling_second(N, K_A)
    denom_B = stirling_second(N, K_B)

    if denom_A == 0 or denom_B == 0:
        return 0.0

    e_joint_entropy = 0.0

    # Iterate all possible cluster sizes 'a' for clustering A
    for a in range(1, N - (K_A - 1) + 1):
        weight_a = (comb(N, a) * stirling_second(N - a, K_A - 1)) / denom_A

        # Iterate all possible cluster sizes 'b' for clustering B
        for b in range(1, N - (K_B - 1) + 1):
            weight_b = (comb(N, b) * stirling_second(N - b, K_B - 1)) / denom_B

            # Iterate overlap 'n' [max(1, a+b-N), min(a, b)]
            start_n = max(1, a + b - N)
            end_n = min(a, b)

            if start_n > end_n:
                continue

            n_values = np.arange(start_n, end_n + 1)

            # Hypergeometric Prob P(n | N, a, b)
            probs_n = hypergeom.pmf(n_values, N, a, b)

            # Joint entropy term: (n/N) * log(n/N)
            term_n = (n_values / N) * np.log(n_values / N)

            expected_term_n = np.sum(term_n * probs_n)

            # Accumulate
            e_joint_entropy += weight_a * weight_b * expected_term_n

    return -e_joint_entropy

#M_num E[MI]
#two-sided
def expected_mi_num_twosided(labels1, labels2):
    """
        Calculate Expected MI under M_num (Two-sided).
    """
    e_ha = expected_entropy_num(labels1)

    e_hb = expected_entropy_num(labels2)

    e_hab = expected_joint_entropy_num(labels1, labels2)

    e_mi = e_ha + e_hb - e_hab

    return e_mi

#M_num E_num[H(A, B)]
#one-sided
def expected_joint_entropy_num_onesided(labels1, labels2):
    """
    Calculate One-sided Expected Joint Entropy E_num^1[H(A, G)] under M_num.

     E^1[H(A, G)] = - sum_{a} Weight(a) * sum_{j} sum_{n} P(n|a, g_j) * (n/N)*log(n/N)

    parameter:
    labels1: Labels for random cluster A (assuming they are drawn from M_num, with N and K_A provided).
    labels2:  Labels for a fixed reference cluster G (providing a fixed cluster structure g_j)
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    N = len(labels1)
    n_ij = contingency_table(labels1, labels2)
    a_i, b_j, _ = marginal_sums(n_ij)


    K_A = len(a_i)

    g_sizes = b_j

    denom_stirling = stirling_second(N, K_A)
    if denom_stirling == 0: return 0.0

    e_joint_entropy = 0.0

    # Iterate possible sizes 'a' for Random Clustering A
    for a in range(1, N - (K_A - 1) + 1):

        # Weight_a = C(N, a) * S(N-a, K_A-1) / S(N, K_A)
        weight_numerator = math.comb(N, a) * stirling_second(N - a, K_A - 1)
        weight_a = weight_numerator / denom_stirling

        if weight_a == 0: continue

        # Iterate fixed clusters g_j in Reference G
        for g_j in g_sizes:
            # Iterate overlap n
            # range: [max(1, a + g_j - N), min(a, g_j)]
            start_n = max(1, a + g_j - N)
            end_n = min(a, g_j)

            if start_n > end_n:
                continue

            n_values = np.arange(start_n, end_n + 1)

            probs_n = hypergeom.pmf(n_values, N, a, g_j)

            term_n = (n_values / N) * np.log(n_values / N)

            expected_term_n = np.sum(term_n * probs_n)

            e_joint_entropy += weight_a * expected_term_n

    return -e_joint_entropy

#M_num E[MI]
#one-sided
def expected_mi_num_onesided(labels1, labels2):
    """
        Calculate One-sided Expected MI under M_num.

        E^1[MI] = E_num[H(A)] + H(G) - E_num^1[H(A, G)]

    """

    e_ha = expected_entropy_num(labels1)

    h_g = entropy_fixed(labels2)

    e_hag = expected_joint_entropy_num_onesided(labels1, labels2)

    e_mi = e_ha + h_g - e_hag

    return e_mi

#M_all E[H(A)]
#two-sided=one-sided
def expected_entropy_all(labels):
    """
    Calculate Expected Entropy E[H(A)] under M_all.
    """
    N = len(labels)
    if N == 0:
        return 0.0

    bell_nums = bell_numbers_list(N)
    BN = bell_nums[N]

    total_entropy = 0.0

    # Formula: - sum_{k=1}^{N} [ binom(N, k) * (B_{N-k} / B_N) * (k/N) * log(k/N) ]
    for k in range(1, N + 1):
        prob_weight = (math.comb(N, k) * bell_nums[N - k]) / BN
        term = (k / N) * math.log(k / N)
        total_entropy += prob_weight * term

    return -total_entropy

#M_all E[H(A, B)]
#two-sided
def expected_joint_entropy_all_twosided(labels1, labels2):
    """
    Calculate Expected Joint Entropy E[H(A, B)] under M_all (Two-sided).
    """
    N = len(labels1)
    if N == 0:
        return 0.0

    bell_nums = bell_numbers_list(N)
    BN = bell_nums[N]

    total_joint_entropy = 0.0

    # Outer loop k: Cluster size in A
    for k in range(1, N + 1):
        weight_k = (math.comb(N, k) * bell_nums[N - k]) / BN

        sum_m = 0.0
        # Middle loop m: Cluster size in B
        for m in range(1, N + 1):
            weight_m = (math.comb(N, m) * bell_nums[N - m]) / BN

            start_n = max(1, k + m - N)
            end_n = min(k, m)

            sum_n = 0.0
            if start_n <= end_n:
                for n in range(start_n, end_n + 1):

                    hyper_num = math.comb(m, n) * math.comb(N - m, k - n)
                    hyper_den = math.comb(N, k)
                    prob_n = hyper_num / hyper_den

                    term_n = (n / N) * math.log(n / N)
                    sum_n += prob_n * term_n

            sum_m += weight_m * sum_n

        total_joint_entropy += weight_k * sum_m

    return -total_joint_entropy

#M_all E[MI]
#two-sided
def expected_mi_all_twosided(labels1, labels2):
    """
    Calculate Expected MI under M_all (Two-sided).

    The model assumes that both clusters are drawn uniformly and randomly from all possible partitions.
    The calculation result depends only on the total number of data points N.

    parameter:
    labels1 -- The first cluster's label list
    labels2 -- The second cluster's label list (must be the same length as labels1).

    return:e_mi
    """
    e_ha = expected_entropy_all(labels1)

    e_hb = expected_entropy_all(labels2)

    e_hab = expected_joint_entropy_all_twosided(labels1, labels2)

    e_mi = e_ha + e_hb - e_hab

    return e_mi

#M_all E[H(A, G)]
#one-sided
def expected_joint_entropy_all_onesided(labels1, labels2):
    """
    Calculate One-sided Expected Joint Entropy E[H(A, G)] under M_all.

    labels1: Random A (provides N)
    labels2: Fixed G
    """
    N = len(labels1)
    if N == 0:
        return 0.0

    # Get cluster size distribution of Reference G
    g_counts = list(Counter(labels2).values())

    bell_nums = bell_numbers_list(N)
    BN = bell_nums[N]

    total_joint_entropy = 0.0

    # Outer loop k: Potential cluster size in Random A
    for k in range(1, N + 1):

        term_k_coeff = bell_nums[N - k] / BN

        # Inner loop: Iterate every cluster g_j in Fixed G
        for g_j in g_counts:

            start_n = max(1, k + g_j - N)
            end_n = min(k, g_j)

            sum_n = 0.0
            if start_n <= end_n:
                for n in range(start_n, end_n + 1):

                    weight_n = math.comb(g_j, n) * math.comb(N - g_j, k - n)

                    term_log = (n / N) * math.log(n / N)

                    sum_n += weight_n * term_log

            total_joint_entropy += term_k_coeff * sum_n

    return -total_joint_entropy

#M_all E[MI]
#one-sided
def expected_mi_all_onesided(labels1, labels2):
    """
    Calculate One-sided Expected MI under M_all.

    E[MI] = E[H(A)] + H(G) - E[H(A, G)]
    """
    e_ha = expected_entropy_all(labels1)

    h_g = entropy_fixed(labels2)

    e_hag = expected_joint_entropy_all_onesided(labels1, labels2)

    e_mi = e_ha + h_g - e_hag

    return e_mi