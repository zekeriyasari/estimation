import numpy as np
import numbers
import functools
from estimation.defaults import *


def numerical_gradient(f, x, h=DERIVSTEP, *args, **kwargs):
    """
    Estimate numerical gradient
    :param f: callable,
        function whose derivative will be taken
    :param x: numbers.Number, np.ndarray,
        point at which the derivative will be taken
    :param args: tuple,
        f arguments
    :param h: dict,
        f keyword arguments
    :return grad: float, np.ndarray,
        numerical gradient of  f at x
    """

    if isinstance(x, numbers.Number):
        n = 1
    elif isinstance(x, np.ndarray):
        n = x.size
    else:
        raise TypeError("Expected type np.ndarray or number.Number, got {}".format(type(x)))

    ff = functools.partial(f, *args, **kwargs)
    fval = ff(x)
    if isinstance(fval, numbers.Number):
        m = 1
    elif isinstance(fval, np.ndarray):
        m = x.size
    else:
        raise TypeError("Expected type np.ndarray or number.Number, got {}".format(type(x)))

    if m == 1:
        if n == 1:
            grad = (ff(x + h) - ff(x - h)) / (2 * h)
            return grad
        elif n > 1:
            grad = np.zeros(n)
            for i in range(n):
                xb = np.copy(x)
                xf = np.copy(x)
                xf[i] = x[i] + h
                xb[i] = x[i] - h
                grad[i] = (ff(xf) - ff(xb)) / (2 * h)
            return grad
    else:
        grad = np.zeros((n, m))
        for i in range(n):
            xb = np.copy(x)
            xf = np.copy(x)
            xf[i] = x[i] + h
            xb[i] = x[i] - h
            for j in range(m):
                grad[i, j] = (ff(xf)[j] - ff(xb)[j]) / (2 * h)
        return grad


def newton_raphson(f, x0, fprime=None, tol=TOL, maxiter=MAXITER, callback=None, *args, **kwargs):
    """
    Newton-Raphson method for root finding in the form:
        f(x) = 0,   f: R^n -> R^m
    :param f: callable,
        function whose root will be calculated
    :param x0: numbers.Number, np.ndarray
        initial guess
    :param fprime: callable,
        derivative function of f. If not provided, it is calculated numerically.
    :param tol: float,
        stopping tolerance
    :param maxiter: int,    
        maximum number of iterations
    :param callback: callable,
        callback function
    :param args: tuple,
        arguments of f
    :param kwargs: 
        keyword arguments of f
    :return: numbers.Number, np.ndarray,
        root of f(x) = 0
    """
    # Check for default parameters
    if tol <= 0:
        raise ValueError("tol must be greater than zero")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than zero")

    # Wrap function if fprime is not given, use numerical calculation
    ff = functools.partial(f, *args, **kwargs)
    if fprime is None:
        def fprime(x):
            return numerical_gradient(f, x)

    p0 = x0
    for i in range(maxiter):
        fder = fprime(p0)
        fval = ff(p0)
        if isinstance(x0, numbers.Number):  # Scalar Newton-Raphson
            p = p0 - fval / fder
        else:
            p = p0 - (np.linalg.inv(fder.dot(fder.T)).dot(fder)).dot(fval).flatten()  # Multivariate Newton-Raphson
        if np.linalg.norm(p - p0) < tol:
            return i, p

        p0 = p
        if callback is not None:
            callback()


def scoring(f, x0, fisher=None, tol=TOL, maxiter=MAXITER, callback=None, *args, **kwargs):
    """
    Newton-Raphson method for root finding in the form:
        f(x) = 0,   f: R^n -> R^m
    :param f: callable,
        function whose root will be calculated
    :param x0: numbers.Number, np.ndarray
        initial guess
    :param fisher: callable,
        Fisher information function of f. If not provided, it is calculated numerically.
    :param tol: float,
        stopping tolerance
    :param maxiter: int,    
        maximum number of iterations
    :param callback: callable,
        callback function
    :param args: tuple,
        arguments of f
    :param kwargs: 
        keyword arguments of f
    :return: numbers.Number, np.ndarray,
        root of f(x) = 0
    """
    # Check for default parameters
    if tol <= 0:
        raise ValueError("tol must be greater than zero")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than zero")

    # Wrap function if fprime is not given, use numerical calculation
    ff = functools.partial(f, *args, **kwargs)
    if fisher is None:
        def fisher(x):
            return numerical_gradient(f, x)

    p0 = x0
    for i in range(maxiter):
        finfo = fisher(p0)
        fval = ff(p0)
        if isinstance(x0, numbers.Number):  # Scalar scoring
            p = p0 - fval / finfo
        else:
            p = p0 - np.linalg.inv(finfo).dot(fval).flatten()  # Multivariate scoring
        if np.linalg.norm(p - p0) < tol:
            return i, p
        p0 = p

        if callback is not None:
            callback()


if __name__ == '__main__':
    pass
