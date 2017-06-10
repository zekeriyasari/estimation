# Example 7.11 in the book
# Exponential in WGN. Estimation problem is
#   x[n] = r ** n + w[n],   n = 0, ..., N - 1
# where r to be estimated.

from estimation.numerical_estimators import *
from matplotlib import pyplot as plt

n = 50  # Number of data samples
r = 0.5  # Exponential true value
variance = 0.01  # Noise variance

nn = np.arange(n)
s = r ** nn  # Signal
w = np.sqrt(variance) * np.random.randn(n)  # Noise
x = s + w  # Observed data


def g(theta):
    return np.sum([(x[k] - theta ** k) * k * theta ** (k - 1) for k in nn])


def dg(theta):
    return np.sum([k * theta ** (k - 2) * ((k - 1) * x[k] - (2 * k - 1) * theta ** k) for k in nn])


r0 = 1.5
niter, root = newton_raphson(g, r0, fprime=dg)
# niter, root = newton_raphson(g, r0)
# niter, root = scoring(g, r0)
print("True value: {}\nEstimated value: {}\nNumber of iterations: {}".format(r, root, niter))

plt.plot(s, label="Signal without noise")
plt.plot(x, label="Signal with noise")
plt.xlabel("Samples [n]")
plt.legend()
plt.show()
