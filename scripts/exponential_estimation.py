from estimation.numerical_estimators import *
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import stats

# Construct signal
l = 50
r = 0.5
n = np.arange(l)
s = r ** n

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
        return np.linalg.norm(x - theta ** n) ** 2


    # Gradient of cost function
    def dj(theta):
        return np.sum(x * n * theta ** (n - 1) - n * theta ** (2 * n - 1))


    # Hessian of cost function
    def ddj(theta):
        return np.sum(x * n * (n - 1) * theta ** (2 * n - 2) - n * (2 * n - 1) * theta ** (2 * n - 2))

    # # Plot cost function
    # dmn = np.arange(0, 1, 0.001)
    # rng = np.array(list(map(j, dmn)))
    # plt.plot(dmn, rng)
    # plt.xlabel("$r$")
    # plt.ylabel("$J(r)$")
    # plt.axvline(r, linestyle='dashed', color="red")
    # plt.show()

    # Minimize cost function
    x0 = np.array([1.])
    theta_min = optimize.minimize(j, x0)['x']

    # Find root by scipy
    root0 = optimize.root(dj, x0)['x']

    # Find root by newton_raphson
    niter, root1 = newton_raphson(dj, x0[0], fprime=ddj, disp=False)

    # print("True value: {}\nScipy optimize: {}\nScipy root: {}\nNewton raphson root: {}".
    #       format(phi0, theta_min, root0, root1))
    theta_hat[i] = root1

print("True Value: {}\nEstimator mean: {}\nEstimator Variance: {}".format(r, theta_hat.mean(), theta_hat.var()))

plt.figure()
pdf = stats.gaussian_kde(theta_hat)  # Gaussian kernel density estimation
dom = np.arange(0., np.pi, 0.01)
plt.plot(dom, pdf(dom))
plt.xlabel("$\hat{r}$")
plt.ylabel("$p(\hat{r})$")
plt.title("$KDE \; of \; \hat{r}$")

plt.figure()
plt.hist(theta_hat)  # Histogram
plt.xlabel("$\hat{r}$")
plt.ylabel("$p(\hat{r})$")
plt.title("$Histogram \; of \; \hat{r}$")
plt.show()
