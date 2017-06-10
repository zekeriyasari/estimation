from unittest import TestCase
from estimation.numerical_estimators import *


class TestNumericalEstimators(TestCase):
    def test_estimate_gradient(self):
        print("Test numerical_gradient")

        def f1(x, b=0.):
            return x ** 2 + 2 * x + b

        def f2(x):
            return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]

        def f3(x):
            return np.array([
                [x[0] ** 2 + 2 * x[0] * x[1]],
                [x[1] ** 2 + 2 * x[0] * x[1]]
            ])

        # Test for n = 1, m = 1
        x0 = 1.0
        f1_prime = numerical_gradient(f1, x0, b=10.)
        self.failUnless(np.isclose(np.array([f1_prime]), np.array([2 * x0 + 2])))

        # Test for n = 2, m = 1
        x0 = np.array([1., 1.])
        f2_prime = numerical_gradient(f2, x0)
        self.failUnless(np.allclose(f2_prime, np.array([2 * x0[0] + 2 * x0[1],
                                                        2 * x0[1] + 2 * x0[0]])))

        # Test for n = 2, m = 2
        x0 = np.array([1., 2.])
        f3_prime = numerical_gradient(f3, x0)
        self.failUnless(np.allclose(f3_prime, np.array([[2 * x0[0] + 2 * x0[1], 2 * x0[1]],
                                                        [2 * x0[0], 2 * x0[1] + 2 * x0[0]]])))
        print("ok...")

    def test_newton_raphson(self):
        print("Test newton_raphson")

        # Test for n=1 m=1
        def f1(x):
            return x ** 3 - x - 1

        x0 = 1.5
        i, root = newton_raphson(f1, x0)
        self.failUnless(np.isclose(np.array([root]), np.array([1.324717957])))

        def f2(x):
            return np.array([
                [3 * x[0] - np.cos(x[1] * x[2]) - 3 / 2],
                [4 * x[0] ** 2 - 625 * x[1] ** 2 + 2 * x[2] - 1],
                [20 * x[2] + np.exp(-(x[0] * x[1])) + 9]
            ])

        x0 = np.array([1., 1., 1.])
        i, root = newton_raphson(f2, x0)
        self.failUnless(np.allclose(root, np.array([0.833282, 0.035335, -0.498549]), rtol=1e-3))

        print("ok...")


