from estimation.numerical_estimators import *

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

# Monte-Carlo simulation
m = 1000  # Number of Monte-Carlo trials
roots = np.array([])
print("Phase estimation - True Value: {}".format(phi))
for i in range(m):
    # phi0 = phi * 1.5  # Initial condition
    phi0 = np.random.normal(loc=phi, scale=0.1)  # Initial condition

    # Noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed Signal
    x = s + w

    # Nonlinear function
    def g(theta):
        return np.sum(
            [(x[k] - a * np.cos(2 * np.pi * f0 * k + theta)) * np.sin(2 * np.pi * f0 * k + theta) for k in range(l)])


    niter, root = newton_raphson(g, phi0, disp=False)
    roots = np.append(roots, root)
    print("Trial-{}: Initial: {}, Estimated Value: {}".format(i, phi0, root))

print("\nTrue Value: {}\nMean: {}\nVariance: {}".format(phi, roots.mean(), roots.var()))
