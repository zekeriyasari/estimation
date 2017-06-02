from unittest import TestCase
from estimation.theoretical_estimators import *


class TestEstimators(TestCase):
    def test_dc_level_estimator(self):
        print("Test dc_level_estimator...")
        a = 1  # Dc level

        # Test for single sample single Monte-Carlo trial
        w = np.random.rand()
        x = a + w
        a_hat = dc_level_estimator(x)
        self.failUnless(a_hat == x)
        print("True value: {}\nEstimated value: {}".format(a, a_hat))

        # Test for multiple data samples single Monte-Carlo trial
        n = 10  # Number of data samples
        w = np.random.randn(n)
        x = a + w
        a_hat = dc_level_estimator(x)
        self.failUnless(a_hat == x.mean())
        print("True value: {}\nEstimated value: {}".format(a, a_hat))

        # Test for multiple data samples multiple Monte-Carlo trial
        m = 5  # Number of Monte-Carlo trials
        n = 10  # Number of data samples
        w = np.random.randn(m, n)
        x = a + w
        a_hat = dc_level_estimator(x)
        self.failUnless(np.allclose(a_hat, x.mean(axis=1)))
        print("True value: {}\nEstimated values: {}".format(a, a_hat))

        # Test for exception
        with self.assertRaises(TypeError):
            x = dict(a='1', b='2')
            a_hat = dc_level_estimator(x)

        print("ok...")

    def test_linear_estimator(self):
        print("Test linear_estimator")
        p = 3  # Number of unknown
        n = 10  # Number of data samples
        theta = np.array([1, 2, 3])  # Unknown parameters
        h = np.zeros((n, p))  # Observation matrix
        for i in range(p):
            h[:, i] = np.array([j ** i for j in range(n)]).T

        # Test for single Monte-Carlo trial
        w = np.random.randn(n)  # Noise
        x = h.dot(theta) + w  # Observed data
        theta_hat = linear_estimator(x, h)
        self.failUnless(np.allclose(theta_hat, np.linalg.inv(h.T.dot(h)).dot(h.T.dot(x))))
        print("True value: {}\nEstimated value: {}".format(theta, theta_hat))

        # Test for multiple Monte-Carlo trial
        m = 10  # Number of Monte-Carlo trials
        w = np.random.randn(m, n)
        x = np.zeros((m, n))
        for j in range(m):
            x[j] = h.dot(theta) + w[j]
        theta_hat = linear_estimator(x, h)
        print("True value: {}\nEstimated value: {}".format(theta, theta_hat))
        print("ok...")

    def test_phase_estimator(self):
        print("Test phase estimator")
        a = 1
        f0 = 0.08
        phi = np.pi / 4
        variance = 0.05

        n = 50  # Number of data samples
        nn = np.arange(n)
        s = a * np.cos(2 * np.pi * f0 * nn + phi)  # Sinusoid

        # Test for single Monte-Carlo trials
        w = np.sqrt(variance) * np.random.randn(n)  # Noise
        x = s + w  # Observed data
        nominator = x.dot(np.sin(2 * np.pi * f0 * nn))
        denominator = x.dot(np.cos(2 * np.pi * f0 * nn))
        phi_hat = phase_estimator(x, f0)
        self.failUnless(phi_hat == -np.arctan(nominator / denominator))

        # Test for multiple Monte-Carlo trials
        m = 5000  # Number of Monte-Carlo trials
        n = np.arange(20, 100, 20)  # Number of data samples
        ms = np.array([])
        vs = np.array([])
        for k in n:
            w = np.sqrt(variance) * np.random.randn(m, k)  # Noise
            nn = np.arange(k)
            s = a * np.cos(2 * np.pi * f0 * nn + phi)  # Sinusoid
            x = s + w  # Observed data
            phi_hat = phase_estimator(x, f0)
            phi_hat_exp = np.zeros(m)
            for i in range(m):
                nominator = x[i, :].dot(np.sin(2 * np.pi * f0 * nn))
                denominator = x[i, :].dot(np.cos(2 * np.pi * f0 * nn))
                phi_hat_exp[i] = -np.arctan(nominator / denominator)
            self.failUnless(np.allclose(phi_hat, phi_hat_exp))

            ms = np.append(ms, phi_hat.mean())
            vs = np.append(vs, k * phi_hat.var())
        print("True value of phi_var.mean(): {}\n"
              "True value of phi_var.var(): {}".format(0.785, 0.1))
        print("Results for {} Monte-Carlo trials\n"
              "phi_var.mean(): {}\n"
              "phi_var.var(): {}".format(m, ms, vs))
