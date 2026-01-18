import numpy as np
from scipy.stats import skewnorm
from scipy.stats import t
from scipy.stats import laplace


# Function: Generate noise for noisy data
def gen_noise(n, type = 1, sigma = None):
    if sigma is None:
        sigma = np.identity(n)
    mean = np.zeros(n)
    """
    type 1: normal distribution
    type 2: skew normal distribution
    type 3: t distribution
    type 4: Laplace distribution
    type 5: Estimated Variance
    """
    if type == 1 or type == 5:
        return np.random.multivariate_normal(mean = mean, cov = sigma)
    if type == 2:
        eps =  skewnorm.rvs(a = 10, loc = 0, scale = 1, size = n)
        mean = skewnorm.mean(a=10, loc=0, scale=1)
        std  = skewnorm.std(a=10, loc=0, scale=1)
        eps = (eps - mean)/std
        return eps
    if type == 3:
        eps = t.rvs(df = 20, loc = 0, scale = 1, size = n)
        mean = t.mean(df=20, loc=0, scale=1)
        std  = t.std(df=20, loc=0, scale=1)
        eps = (eps - mean)/std
        return eps
    if type == 4:
        eps = laplace.rvs(loc=0, scale=1, size=n)
        mean = laplace.mean(loc=0, scale=1)
        std = laplace.std(loc = 0, scale=1)
        eps = (eps - mean)/std
        return eps


# Function: Generate synthentic data
def generate_data(N, d, true_beta = None, sigma = None, q_prob = 0.2, delta = 0, noise_type = 1):
    X = np.random.normal(scale = 1, size=(N, d))
    if true_beta is None:
        true_beta = [i%2 + 1 for i in range(d)]
    y = np.dot(X, true_beta) + gen_noise(N, type=noise_type, sigma=sigma)
    outlier_indices = []
    if q_prob != 0:
        num_outliers = int(N*q_prob)
        outlier_indices = np.random.choice(N, num_outliers, replace=False)
        y[outlier_indices] += delta

    y = y.reshape(N, 1)
    return X, y, sorted(outlier_indices)
            
