import numpy as np
import numbers


def dc_level_estimator(x):
    """
    Dc level estimator in WGN
        x[n] = a + w[n],    n = 0, ..., N - 1
    if N = 1
        a_hat = x 
    if N > 1
        a_hat = 1 / N * np.sum(x[n]) 
    where a_hat is mvu estimator of a.
    
    :param x: np.ndarray,
        observed data
    :param axis: int,
        axis over which estimator will perform
    :return a_hat: float, ndarray
        estimation of Dc level
    """
    if isinstance(x, numbers.Number):
        a_hat = x
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            a_hat = x.mean()
        else:
            a_hat = x.mean(axis=1)
    else:
        raise TypeError('Expected type is np.ndarray or numbers.Number, got {}'.format(type(x)))
    return a_hat


def linear_estimator(x, h):
    """
    Linear model parameter estimator in WGN
        x = h * theta + w,    
    theta_hat = (h'h)^(-1)h'x 
    where theta_hat is the mvu estimator of theta.
    
    :param x: np.ndarray,
        m-by-n observed data
    :param h: np.ndarray,
        n-by-p observation matrix
    :return theta-hat: np.ndarray,
        m-by-p estimation of linear model parameters
    """
    if x.ndim == 1:
        theta_hat = np.linalg.inv(h.T.dot(h)).dot(h.T.dot(x))
    else:
        m = x.shape[0]
        p = np.linalg.inv(h.T.dot(h)).dot(h.T.dot(x[0, :])).size
        theta_hat = np.zeros((m, p))
        for i in range(m):
            theta_hat[i] = np.linalg.inv(h.T.dot(h)).dot(h.T.dot(x[i, :]))
    return theta_hat


def phase_estimator(x, f0):
    """
    Sinusoidal phase estimator in WGN
        x[n] = a * cos(2 * pi * f0 * n + phi) + w[n]    n = 0, ..., N - 1 
    phi_hat = arctan(sum(x[n] * sin(2 * pi * f * n)) / sum(x[n] * cos(2 * pi * f * n)))
    where phi_hat is the MLE(Maximum Likelihood Estimator) of theta
    
    :param x: np.ndarray,
        observed data
    :param f0: float,
        normalized frequency of data
    :return phi_hat: float, ndarray
        MLE of phi
    """
    if x.ndim == 1:
        n = x.size
        nn = np.arange(n)
        nominator = x.dot(np.sin(2 * np.pi * f0 * nn))
        denominator = x.dot(np.cos(2 * np.pi * f0 * nn))
        phi_hat = -np.arctan(nominator / denominator)
    else:
        m, n = x.shape
        nn = np.arange(n)
        phi_hat = np.zeros(m)
        for i in range(m):
            nominator = x[i, :].dot(np.sin(2 * np.pi * f0 * nn))
            denominator = x[i, :].dot(np.cos(2 * np.pi * f0 * nn))
            phi_hat[i] = -np.arctan(nominator / denominator)
    return phi_hat
