from estimation.numerical_estimators import *
from scipy.signal import chirp
from matplotlib import pyplot as plt

# TODO: Convert continuous-time and discrete-time parameters...

# Sampler parameters
nn = 100  # Number of data samples
fs = 100  # Sampling frequency
ts = 1 / fs

# Chirp signal parameters
a = 1.0  # Amplitude
phi = 0  # Phase
f0 = 1
f1 = 10
t1 = 1
mu = (f1 - f0) / t1  # Chirp rate
w0 = 2 * np.pi * f0  # Carrier frequency

# Noise parameters
variance = 0.01  # Variance

n = np.arange(nn)
t = n * ts
s = chirp(t, f0, t1, f1, phi=phi)  # Chirp signal
w = np.sqrt(variance) * np.random.randn(nn)  # Noise
x = s + w  # Observed signal


def g(theta):
    return np.sum(
        [(x[k] - a * np.cos(w0 * k + theta * np.pi / (fs ** 2) * (k ** 2) + phi)) * np.sin(
            w0 * k + theta * np.pi / (fs ** 2) * (k ** 2) + phi) * k ** 2
         for k in n])


mu0 = 15
# niter, root = newton_raphson(g, r0, fprime=dg)
niter, root = newton_raphson(g, mu0)
# niter, root = scoring(g, r0)
print("True value: {}\nEstimated value: {}\nNumber of iterations: {}".format(mu, root, niter))

plt.plot(s, label="Signal without noise")
plt.plot(x, label="Signal with noise")
plt.legend()
plt.xlabel("n")
plt.show()
