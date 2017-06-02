import numpy as np
import numbers
from estimation.defaults import *


def numerical_gradient(f, x, h=DERIVSTEP):
    if isinstance(x, numbers.Number):
        n = 1
    elif isinstance(x, np.ndarray):
        n = x.size
    else:
        raise TypeError("Expected type np.ndarray or number.Number, got {}".format(type(x)))

    fval = f(x)
    if isinstance(fval, numbers.Number):
        m = 1
    elif isinstance(fval, np.ndarray):
        m = x.size
    else:
        raise TypeError("Expected type np.ndarray or number.Number, got {}".format(type(x)))

    if m == 1:
        if n == 1:
            grad = f(x + h) - f(x - h) / (2 * h)
        elif n > 1:
            grad = np.zeros(n)
            for i in range(n):
                grad[i] = f(x + h)[i] - f(x - h)[i] / (2 * h)
