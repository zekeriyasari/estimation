import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def gaussian_pdf(x, mu=0., sigma2=1.):
    """
    Gaussian pdf function
    :param x: np.ndarray,
        data point(s)
    :param mu: float,
        mean
    :param sigma2: float,
        variance
    """
    return 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-1 / 2 / sigma2 * (x - mu) ** 2)


lengths = np.arange(5, 30, 5)  # Number of data lengths
m = 5000  # Number of Monte-carlo realization
means = np.zeros(lengths.size)  # Estimator means
variances = np.zeros(lengths.size)  # Estimator variances

for i, n in enumerate(lengths):
    a = 1  # DC level signal
    w = np.sqrt(a) * np.random.randn(m, n)  # Noise
    x = a + w  # Observed signal

    a_hat = np.zeros(m)
    for j in range(m):
        a_hat[j] = -1 / 2 + np.sqrt(1 / n * x[j, :].dot(x[j, :]) + 1 / 4)  # Estimated Dc level

    pdf_a_hat = gaussian_kde(a_hat)  # Estimated pdf of estimator

    domain = np.linspace(a_hat.min(), a_hat.max(), 1001)
    plt.plot(domain, pdf_a_hat(domain), label='kde')  # Estimated pdf plot
    plt.plot(domain, gaussian_pdf(domain, mu=a, sigma2=2 / 3 / n), label='theoretical')  # Theoretical pdf plot
    plt.legend()
    plt.savefig('{}'.format(n))
    plt.close()

    means[i] = a_hat.mean()
    variances[i] = a_hat.var() * n
