from estimation.numerical_estimators import *
from matplotlib import pyplot as plt
from scipy.optimize import newton

# Signal
r = 0.5
l = 50
n = np.arange(l)
s = r ** n

# Monte-Carlo simulation
m = 100  # Number of Monte-Carlo trials
not_converged = 0  # Number of un-converged trial
error = np.array([])
print("Phase estimation - True Value: {}".format(r))
print("{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}".format("Trial", "Trial Value", "Initial Value", "Estimated Value",
                                                   "Number of Iterations"))
for i in range(m):
    theta0 = np.random.rand()  # Initial condition

    # Noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed Signal
    x = s + w


    # Nonlinear function
    def j(theta):
        return -np.sum([(x[k] - theta ** k) ** 2 for k in range(l)])


    def dj(theta):
        return np.sum(
            [(x[k] - theta ** k) * k * theta ** (k - 1) for k in range(l)])


    def d2j(theta):
        return np.sum([k * theta ** (k - 2) * ((k - 1) * x[k] - (2 * k - 1) * theta ** k)
                       for k in range(l)])


    try:
        trial_min = newton(lambda x: -dj(x), theta0)
    except RuntimeError:
        pass

    try:
        # niter, root = newton_raphson(dj, phi0, disp=False)
        niter, root = newton_raphson(dj, theta0, fprime=d2j, disp=False)
    except EndOfIteration:
        not_converged += 1
    error = np.append(error, abs(root - trial_min))
    print("{:<20d}{:<20.16f}{:<20.10f}{:<20.16f}{:<20d}".format(i, trial_min, theta0, root, niter))

    step_size = 0.001
    x_line = np.arange(0., 1., step_size)
    y_line = np.array(list(map(j, x_line)))
    plt.plot(x_line, y_line)
    plt.axhline(0.)
    plt.show()

print("\nNumber of Monte-Carlo trials: {}\n"
      "Number of un-converged trials: {}\n"
      "True Value: {}\n"
      "Estimator Mean: {}\n"
      "Estimator Variance: {}\n".format(m, not_converged, r, error.mean(), error.var()))
