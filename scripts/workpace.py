from estimation.numerical_estimators import *
import matplotlib.pyplot as plt

# Sampler
fs = 100

# Signal
l = 50
a = 1.
phi = np.pi / 6
fc = 10
n = np.arange(l)
f0 = fc / fs
s = np.cos(2 * np.pi * f0 * n + phi)  # Signal samples

# Noise
variance = 0.01
w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

# Observed Signal
x = s + w


# Estimation
def g(theta):
    return np.sum(
        [(x[k] - a * np.cos(2 * np.pi * f0 * k + theta)) * np.sin(2 * np.pi * f0 * k + theta) for k in range(l)])


phi0 = phi * 1.5
niter, root = newton_raphson(g, phi0)
print("True Value: {}\nEstimated Value: {}".format(phi, root))

plt.plot(s)
plt.show()
