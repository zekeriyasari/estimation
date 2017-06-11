from estimation.numerical_estimators import *

# Sampler
fs = 100

# Signal
l = 50
a = 1.
phi = 0.
fc = 10
n = np.arange(l)
f0 = fc / fs
s = a * np.cos(2 * np.pi * f0 * n + phi)  # Signal samples

# Monte-Carlo simulation
m = 1000  # Number of Monte-Carlo trials
not_converged = 0  # Number of un-converged trials
roots = np.array([])
print("Phase estimation - True Value: {}".format(f0))
for i in range(m):
    # phi0 = phi * 1.5  # Initial condition
    freq0 = np.random.normal(loc=f0, scale=0.1)  # Initial condition

    # Noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed Signal
    x = s + w

    # Nonlinear function
    def g(theta):
        return np.sum(
            [(x[k] - a * np.cos(2 * np.pi * theta * k + phi)) * np.sin(2 * np.pi * theta * k + phi) * k for k in
             range(l)])


    try:
        niter, root = newton_raphson(g, freq0, disp=False)
    except EndOfIteration:
        not_converged += 1
    roots = np.append(roots, root)
    print("Trial-{}: Initial: {}, Estimated Value: {}".format(i, freq0, root))

print("\nNumber of Monte-Carlo trials: {}\n"
      "Number of un-converged trials: {}\n"
      "True Value: {}\n"
      "Estimator Mean: {}\n"
      "Estimator Variance: {}\n".format(m, not_converged, f0, roots.mean(), roots.var()))
