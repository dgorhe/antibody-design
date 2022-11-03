import numpy as np
from scipy.special import kl_div
from collections import defaultdict
from tqdm import tqdm

def generate_random_subsets(x, sample_size, samples=10):
    subsamples = []
    
    for _ in range(samples):
        subsample = np.random.permutation(x)[:sample_size]
        subsamples.append(subsample)

    return subsamples

def subsampled_kl_divergence(x, y):

    x_subsets = generate_random_subsets(x, sample_size=10, samples=10)
    y_subsets = generate_random_subsets(y, sample_size=10, samples=10)
    try:
        for x in x_subsets:
            assert len(x[x <= 0]) == 0
        for y in y_subsets:
            assert len(y[y <= 0]) == 0
    except AssertionError:
        print('Some subsamples contain 1 or more non-positive values')
        return

    cumulative_kl = 0
    for x_sample in x_subsets:
        for y_sample in y_subsets:
            cumulative_kl += kl_div(x_sample, y_sample).sum()

    return cumulative_kl

def calculate_n_gram(examples: list[str], n=2) -> dict:
    """
    Calculates number of occurances of n characters in a string
    by moving a sliding window of size n across each string.

    Example:
        AAABCDC should return a dictionary as follows:
        {'AA': 2, 'AB': 1, 'BC': 1, 'CD': 1, 'DC': 1}
    """
    n_gram = defaultdict(lambda: 0)
    
    for ex in tqdm(examples):
        for idx in range(len(ex) - n):
            key = ex[idx:idx+n]
            n_gram[key] += 1

    return n_gram


if __name__ == "__main__":
    # # Testing subsampled_k_divergence()
    # x = np.random.normal(2, 1, 20); x = x[x > 0]
    # y = np.random.normal(2, 1, 20); y = y[y > 0]

    # print(x)
    # print(y)
    # print(subsampled_kl_divergence(x, y))

    # Testing calculate_n_gram
    test_examples = ["fhjdsljaf", "rueiworuwe", "fjkdsljfkl;sdj", "hfdlsabv,xcb"]
    dist = calculate_n_gram(test_examples, n=3)
    print(dist)

    
