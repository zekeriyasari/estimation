from estimation.numerical_estimators import *
from scipy import signal
from scipy import stats
from scipy import optimize
from matplotlib import pyplot as plt

# Construct a chirp signal
a = 1.
phi0 = 0.
f0 = 1.
t1 = 1.
f1 = 11.
mu = (f1 - f0) / t1
fs = 100
t = np.arange(0., t1, 1. / fs)
s = a * signal.chirp(t, f0, t1, f1, phi=phi0)
l = s.size
n = np.arange(l)

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
        return np.linalg.norm(x - a * signal.chirp(t, f0, t1, f0 + theta * t1, phi=phi0)) ** 2


    # Gradient of cost function
    def dj(theta):
        return np.sum(n ** 2 *
                      (x * signal.chirp(t, f0, t1, f0 + theta * t1, phi=phi0 - np.pi / 2)
                       - a / 2 * signal.chirp(t, 2 * f0, t1, 2 * (f0 + theta * t1), phi=2 * (phi0 - np.pi / 2))))


    # Hessian of cost function
    def ddj(theta):
        return np.sum(n ** 4 * np.pi / fs ** 2 *
                      (x * signal.chirp(t, f0, t1, f0 + theta * t1, phi=phi0)
                       - a * signal.chirp(t, 2 * f0, t1, 2 * (f0 + theta * t1), phi=2 * phi0)))


    # Plot cost function
    dmn = np.arange(0., 20, 0.01)
    rng = np.array(list(map(j, dmn)))
    plt.plot(dmn, rng)
    plt.show()

    # Minimize cost function
    x0 = np.array([10.5])
    theta_min = optimize.minimize(j, x0)['x']

    # Find root by scipy
    root0 = optimize.root(dj, x0)['x']

    # Find root by newton_raphson
    niter, root1 = newton_raphson(dj, x0[0], disp=False)

    print("True value: {}\nScipy optimize: {}\nScipy root: {}\nNewton raphson root: {}".
          format(mu, theta_min, root0, root1))
    theta_hat[i] = root1

print("\nTrue Value: {}\nEstimator mean: {}\nEstimator Variance: {}".format(mu, theta_hat.mean(), theta_hat.var()))

plt.figure()
pdf = stats.gaussian_kde(theta_hat)  # Gaussian kernel density estimation
dom = np.arange(10., 11., 0.001)
plt.plot(dom, pdf(dom))
plt.xlabel("$\hat{\mu}$")
plt.ylabel("$p(\hat{\mu})$")
plt.title("$KDE \; of \; \hat{\mu}$")

plt.figure()
plt.hist(theta_hat)  # Histogram
plt.xlabel("$\hat{\mu}$")
plt.ylabel("$p(\hat{\mu})$")
plt.title("$Histogram \; of \; \hat{\mu}$")
plt.show()
