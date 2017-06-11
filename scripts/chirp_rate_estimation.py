from estimation.numerical_estimators import *
from scipy.signal import chirp
from matplotlib import pyplot as plt

# Sampler
fs = 100

# Signal
l = 200
a = 1.
phi = 0.
f0 = 1
f1 = 10
t1 = 1.
n = np.arange(l)
t = n / fs
s = chirp(t, f0, t1, f1)  # Signal samples
w0 = 2 * np.pi * f0 / fs  # Discrete frequency
mu = (f1 - f0) / (t1 * fs ** 2) * np.pi  # Discrete chirp rate
# s = a * np.cos(2 * np.pi * f0 * n + phi)  # Signal samples

# Monte-Carlo simulation
m = 100  # Number of Monte-Carlo trials
not_converged = 0  # Number of un-converged trials
roots = np.array([])
print("Phase estimation - True Value: {}".format(mu))
print("{:20s}{:20s}{:20s}{:20s}".format("Trial", "Initial", "Estimated Value", "Number of Iterations"))
for i in range(m):
    # mu0 = mu * 1.5  # Initial condition
    mu0 = np.random.normal(loc=mu, scale=0.001)  # Initial condition

    # Noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed Signal
    x = s + w

    # Nonlinear function
    def g(theta):
        return np.sum(
            [(x[k] - a * np.cos(w0 * k + theta * k ** 2 + phi)) * np.sin(w0 * k + theta * k ** 2 + phi) * k ** 2
             for k in range(l)])


    # Derivative of nonlinear function
    def dg(theta):
        return np.sum([a * np.sin(w0 * k + theta * k ** 2 + phi) ** 2 * k ** 2 +
                       (x[k] - a * np.cos(w0 * k + theta * k ** 2 + phi)) *
                       np.cos(w0 * k + theta * k ** 2 + phi) * k ** 4 for k in range(l)])


    try:
        niter, root = newton_raphson(g, mu0, fprime=dg, disp=False)
    except EndOfIteration:
        not_converged += 1
    roots = np.append(roots, root)
    print("{:<20d}{:<20.16f}{:<20.16f}{:<20d}".format(i, mu0, root, niter))

print("\nNumber of Monte-Carlo trials: {}\n"
      "Number of un-converged trials: {}\n"
      "True Value: {}\n"
      "Estimator Mean: {}\n"
      "Estimator Variance: {}\n".format(m, not_converged, mu, roots.mean(), roots.var()))
