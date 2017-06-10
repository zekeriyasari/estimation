# Sinusoidal in WGN. Estimation problem is
#   x[n] = a * cos(2 * pi * f0 * n + phi) + w[n],   n = 0, ..., N - 1
# where phi to be estimated.

from estimation.numerical_estimators import *
from matplotlib import pyplot as plt

nn = 50  # Number of data samples
f0 = 0.1  # Frequency
phi = np.pi / 2  # Phase in radians
a = 1.  # Amplitude
variance = 0.01  # Variance

n = np.arange(nn)
s = a * np.cos(2 * np.pi * f0 * n + phi)  # Sinusoid
w = np.sqrt(variance) * np.random.randn(nn)  # Noise
x = s + w  # Observed signal


def g(theta):
    return np.sum([(x[k] - a * np.cos(2 * np.pi * f0 * k + theta)) * np.sin(2 * np.pi * f0 * k + theta) for k in n])


phi0 = np.pi / 6
# niter, root = newton_raphson(g, r0, fprime=dg)
niter, root = newton_raphson(g, phi0)
# niter, root = scoring(g, r0)
print("True value: {}\nEstimated value: {}\nNumber of iterations: {}".format(phi, root, niter))

plt.plot(n, s, label="Signal without noise")
plt.plot(n, x, label="Signal with noise")
plt.xlabel("n")
plt.legend()
plt.show()
