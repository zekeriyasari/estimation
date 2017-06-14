from estimation.numerical_estimators import *
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import stats

# Construct signal
l = 50
a = 2.
s = a * np.ones(l)

# Monte-Carlo simulation
m = 1000
theta_hat = np.zeros(m)
for i in range(m):
    # Construct noise
    variance = 0.01
    w = np.sqrt(variance) * np.random.randn(l)

    # Construct observed signal samples
    x = s + w


    # Cost function
    def j(theta):
        return np.linalg.norm(x - theta * np.ones(l)) ** 2


    # Gradient of cost function
    def dj(theta):
        return np.sum(x - theta * np.ones(l))


    # Hessian of cost function
    def ddj(theta):
        return -l


    # Minimize cost function
    x0 = np.array([4.])
    theta_min = optimize.minimize(j, x0)['x']

    # Find root by scipy
    root0 = optimize.root(dj, x0)['x']

    # Find root by newton_raphson
    niter, root1 = newton_raphson(dj, x0[0], fprime=ddj, disp=False)

    # print("True value: {}\nScipy optimize: {}\nScipy root: {}\nNewton raphson root: {}".
    #       format(a, theta_min, root0, root1))
    theta_hat[i] = root1

print("True Value: {}\nEstimator mean: {}\nEstimator Variance: {}".format(a, theta_hat.mean(), theta_hat.var()))

plt.figure()
pdf = stats.gaussian_kde(theta_hat)  # Gaussian kernel density estimation
dom = np.arange(1, 3, 0.01)
plt.plot(dom, pdf(dom))
plt.xlabel("$\hat{A}$")
plt.ylabel("$p(\hat{A})$")
plt.title("$KDE \; of \; \hat{A}$")

plt.figure()
plt.hist(theta_hat)  # Histogram
plt.xlabel("$\hat{A}$")
plt.ylabel("$p(\hat{A})$")
plt.title("$Histogram \; of \; \hat{A}$")
plt.show()
