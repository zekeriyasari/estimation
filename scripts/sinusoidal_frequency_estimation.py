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
s = a * np.cos(2 * np.pi * f0 * n + phi)  # Signal samples

# Noise
variance = 0.02
w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

# Observed Signal
x = s + w


# Nonlinear function
def g(theta):
    return np.sum(
        [(x[k] - a * np.cos(2 * np.pi * theta * k + phi)) * np.sin(2 * np.pi * theta * k + phi) * k for k in range(l)])


f = f0 * 1.2
niter, root = newton_raphson(g, f)
print("True Value: {}\nEstimated Value: {}".format(f0, root))

plt.plot(n, s, "r", label="Signal without noise")
plt.stem(n, x, label="Signal with noise")
plt.legend()
plt.show()
