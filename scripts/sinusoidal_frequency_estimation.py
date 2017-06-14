from estimation.numerical_estimators import *
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import stats

# Construct signal
l = 50
a = 1.
f0 = 0.25
phi0 = np.pi / 6
n = np.arange(l)
s = a * np.cos(2 * np.pi * f0 * n + phi0)

# Monte-Carlo simulation
m = 1000  # Number of Monte-Carlo simulations
theta_hat = np.zeros(m)
for i in range(m):
    # Construct noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)

    # Construct observed signal samples
    x = s + w

    # Cost function
    def j(theta):
        return np.linalg.norm(x - a * np.cos(2 * np.pi * theta * n + phi0)) ** 2


    # Gradient of cost function
    def dj(theta):
        return np.sum(n * (x * np.sin(2 * np.pi * theta * n + phi0) - a / 2 * np.sin(4 * np.pi * f0 * n + 2 * theta)))


    # Hessian of cost function
    def ddj(theta):
        return np.sum(2 * np.pi * n ** 2 *
                      (x * np.cos(2 * np.pi * theta * n + phi0) - a * np.cos(4 * np.pi * f0 * n + 2 * theta)))

    # Plot cost function
    dmn = np.arange(0., 0.5, 0.0001)
    rng = np.array(list(map(j, dmn)))
    plt.plot(dmn, rng)
    plt.xlabel("$f_0$")
    plt.ylabel("$J(f_0)$")
    plt.axvline(f0, linestyle='dashed', color="red")
    plt.show()

    # Minimize cost function
    x0 = np.array([0.4])
    theta_min = optimize.minimize(j, x0)['x']

    # Find root by scipy
    root0 = optimize.root(dj, x0)['x']

    # Find root by newton_raphson
    niter, root1 = newton_raphson(dj, x0[0], fprime=ddj, disp=False)

    # print("True value: {}\nScipy optimize: {}\nScipy root: {}\nNewton raphson root: {}".
    #       format(f0, theta_min, root0, root1))
    theta_hat[i] = root1

print("True Value: {}\nEstimator mean: {}\nEstimator Variance: {}".format(f0, theta_hat.mean(), theta_hat.var()))

plt.figure()
pdf = stats.gaussian_kde(theta_hat)  # Gaussian kernel density estimation
dom = np.arange(0., 0.5, 0.001)
plt.plot(dom, pdf(dom))
plt.xlabel("$\hat{f_0}$")
plt.ylabel("$p(\hat{f_0})$")
plt.title("$KDE \; of \; \hat{f_0}$")

plt.figure()
plt.hist(theta_hat)  # Histogram
plt.xlabel("$\hat{f_0}$")
plt.ylabel("$p(\hat{f_0})$")
plt.title("$Histogram \; of \; \hat{f_0}$")
plt.show()
