import numpy as np
from scipy.special import kl_div

def generate_random_subsets(x, sample_size, samples=10):
    subsamples = []
    
    for _ in range(samples):
        subsample = np.random.permutation(x)[:sample_size]
        subsamples.append(subsample)

    return subsamples

def find_kl_divergence(x, y):
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

def find_n_gram_distribution(examples):
    pass

if __name__ == "__main__":
    x = np.random.normal(2, 1, 20); x = x[x > 0]
    y = np.random.normal(2, 1, 20); y = y[y > 0]

    print(x)
    print(y)

    print(find_kl_divergence(x, y))

    
