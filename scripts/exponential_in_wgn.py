# Example 7.11 in the book
# Exponential in WGN. Estimation problem is
#   x[n] = r ** n + w[n],   n = 0, ..., N - 1
# where r to be estimated.

from estimation.numerical_estimators import *

n = 50  # Number of data samples
r = 0.5  # Exponential true value
variance = 0.01  # Noise variance

nn = np.arange(n)
s = r ** nn  # Signal
w = np.sqrt(variance) * np.random.randn(n)  # Noise
x = s + w  # Observed data


def g(theta):
    return np.sum([(x[k] - theta ** k) * k * theta ** (k - 1) for k in nn])


r0 = 0.2
niter, root = newton_raphson(g, r0)
# niter, root = scoring(g, r0)
print("True value: {}\nEstimated value: {}\nNumber of iterations: {}".format(r, root, niter))
