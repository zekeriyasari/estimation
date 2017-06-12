from estimation.numerical_estimators import *
from matplotlib import pyplot as plt
from scipy.optimize import newton

# Sampler
fs = 100

# Signal
l = 50
a = 1.
phi = np.pi / 6
fc = 10
n = np.arange(l)
f0 = fc / fs
w0 = 2 * np.pi * f0
s = a * np.cos(w0 * n + phi)  # Signal samples

# Monte-Carlo simulation
m = 100  # Number of Monte-Carlo trials
not_converged = 0  # Number of un-converged trial
error = np.array([])
print("Phase estimation - True Value: {}".format(phi))
print("{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}".format("Trial", "Trial Value", "Initial Value", "Estimated Value",
                                                          "Theoretical Value", "Number of Iterations"))
for i in range(m):
    # phi0 = phi * 1.25  # Initial condition
    phi0 = np.random.rand() * np.pi / 2  # Initial condition

    # Noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed Signal
    x = s + w


    # Nonlinear function
    def g(theta):
        return np.sum([(x[k] - a * np.cos(w0 * k + theta)) ** 2 for k in range(l)])


    def dg(theta):
        return np.sum(
            [(x[k] - a * np.cos(w0 * k + theta)) * np.sin(w0 * k + theta) for k in range(l)])


    def d2g(theta):
        return np.sum([a * np.sin(w0 * k + theta) ** 2 + (x[k] - a * np.cos(w0 * k + theta)) * np.cos(w0 * k + theta)
                       for k in range(l)])


    # Scipy root finding
    trial_min = newton(dg, phi0)

    # Newton-Raphson root finding
    try:
        niter, root = newton_raphson(dg, phi0, disp=False)
        # niter, root = newton_raphson(dg, phi0, fprime=d2g, disp=True)
    except EndOfIteration:
        not_converged += 1
    # error = np.append(error, abs(root - trial_min))

    # Theoretical estimation
    num = x.dot(np.sin(w0 * n))
    denum = x.dot(np.cos(w0 * n))
    phi_hat = -np.arctan(num / denum)
    print("{:<20d}{:<20.16f}{:<20.10f}{:<20.16f}{:<20.16f}{:<20d}".format(i, trial_min, phi0, root, phi_hat, niter))

    # step_size = 0.001
    # x_line = np.arange(0., 2 * np.pi, step_size)
    # y_line = np.array(list(map(dg, x_line)))
    # plt.plot(x_line, y_line)
    # plt.axhline(0.)
    # plt.show()

# print("\nNumber of Monte-Carlo trials: {}\n"
#       "Number of un-converged trials: {}\n"
#       "True Value: {}\n"
#       "Estimator Mean: {}\n"
#       "Estimator Variance: {}\n".format(m, not_converged, phi, error.mean(), error.var()))
