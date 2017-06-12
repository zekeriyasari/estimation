from estimation.numerical_estimators import *
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Construct signal samples
l = 50  # Number of data points
a = 1.  # Amplitude
f0 = 0.25  # Frequency
phi = np.pi / 6  # Phase
n = np.arange(l)
w0 = 2 * np.pi * f0
s = a * np.cos(w0 * n + phi)  # Signal samples

# Monte-Carlo Simulation
m = 100  # Number of Monte-Carlo simulation
print("True Value: {}".format(np.array([f0, phi])))
for i in range(m):
    # Construct noise
    variance = 0.1  # Noise variance
    w = np.sqrt(variance) * np.random.randn(l)  # Noise samples

    # Observed samples
    x = s + w
    x = s

    # Cost function
    def j(theta):
        f0, phi = theta
        w0 = 2 * np.pi * f0
        return np.sum([(x[k] - a * np.cos(w0 * k + phi)) ** 2 for k in range(l)])


    # Gradient vector
    def jac(theta):
        f0, phi = theta
        w0 = 2 * np.pi * f0
        return np.array([
            4 * np.pi * a * np.sum([(x[k] - a * np.cos(w0 * k + phi)) * np.sin(w0 * k + phi) * k for k in range(l)]),
            2 * a * np.sum([(x[k] - a * np.cos(w0 * k + phi)) * np.sin(w0 * k + phi) for k in range(l)])
        ])


    # Hessian matrix
    def hes(theta):
        f0, phi = theta
        w0 = 2 * np.pi * f0
        j11 = -4 * np.pi ** 2 * a / variance * np.sum([a * k ** 2 * np.sin(w0 * k + phi) ** 2 +
                                                       (x[k] - a * np.cos(w0 * k + phi)) * k ** 2 * np.cos(w0 * k + phi)
                                                       for k in range(l)])
        j12 = -2 * np.pi * a / variance * np.sum([a * k * np.sin(w0 * k + phi) ** 2 +
                                                  (x[k] - a * np.cos(w0 * k + phi)) * k * np.cos(w0 * k + phi)
                                                  for k in range(l)])
        j22 = -a / variance * np.sum([a * np.sin(w0 * k + phi) ** 2 +
                                      (x[k] - a * np.cos(w0 * k + phi)) * np.cos(w0 * k + phi)
                                      for k in range(l)])
        return np.array([
            [j11, j12],
            [j12, j22]
        ])


    x0 = np.array([0.125, np.pi / 2])

    # scipy.optimize.root
    trial_min = optimize.root(jac, x0)['x']

    # Newton-Raphson root
    niter, root = newton_raphson(jac, x0, disp=False)

    # Theoretical estimation
    # Estimate frequency
    def prdgrm(f):
        kernel = np.exp(-2j * np.pi * f * n)
        return 1 / l * np.abs(x.dot(kernel)) ** 2


    step = 0.001
    x_line = np.arange(0., 0.5, step)
    y_line = np.array(list(map(prdgrm, x_line)))
    f0_hat = np.argmax(y_line) * step
    # f0_hat = optimize.minimize(prdgrm, x0[1])['x']
    # print(f0_hat)
    # plt.plot(x_line, y_line)
    # plt.show()

    # Estimate phase
    num = -x.dot(np.sin(2 * np.pi * f0_hat * n))
    denum = x.dot(np.cos(2 * np.pi * f0_hat * n))
    phi_hat = np.arctan(num / denum)

    # Print the results
    print("Iteration: ", i, "Scipy Root: ", trial_min, "Newton_Rapshon: ", root, "Theoretical Freq", f0_hat,
          "Theoretical Phase", phi_hat)

    # print("Theoretical Value: {}".format(phi_hat))


