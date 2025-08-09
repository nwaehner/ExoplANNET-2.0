import numpy as np


class Kernel(object):
    """
    Base class for Gaussian process kernels.

    :param array-like alpha: The hyperparameters.
    """
    def __init__(self, alpha):
        """
        :param array-like alpha: The hyperparameters.
        """
        self.hyperparams = alpha

    def covariance(self, dx):
        """
        :param np.array dx: The difference matrix. This can be a numpy array of
                            arbitrary shape.

        :return kermatrix: The kernel matrix (same shape as the input ``dx``).
        """
        # Perform checks on dx
        # Return
        return self._covariance(dx)

    def _covariance(self, dx):
        raise NotImplemented('_covariance method should be implemented by '
                             'subclass.')

    def sample(self, x, size=1):
        dx = x[:, None] - x[None, :]
        return np.random.multivariate_normal(np.zeros_like(x),
                                             self.covariance(dx),
                                             size)


class SquaredExponentialKernel(Kernel):
    """
    A class implementing the squared exponential kernel.

    :param array-like alpha: (2,) The parameter vector (amplitude,
    lengthscale).
    """
    def __init__(self, alpha):
        """
        :param array-like alpha: (2,) The parameter vector (amplitude,
        lengthscale).
        """
        super(SquaredExponentialKernel, self).__init__(alpha)

    def _covariance(self, dx):
        alpha = self.hyperparams
        return alpha[0] ** 2 * np.exp(-0.5 * dx ** 2 / alpha[1] ** 2)


class GeneralisedExponentialKernel(Kernel):
    """
    A class implementing the squared exponential kernel.

    :param array-like alpha: (3,) The parameter vector (amplitude,
    lengthscale, exponent).
    """
    def __init__(self, alpha):
        """
        :param array-like alpha: (3,) The parameter vector (amplitude,
        lengthscale, exponent).
        """
        super(GeneralisedExponentialKernel, self).__init__(alpha)

    def _covariance(self, dx):
        alpha = self.hyperparams
        return alpha[0] ** 2 * np.exp(-(0.5 * np.abs(dx) / alpha[1])**alpha[2])


class QuasiPeriodicKernel(Kernel):
    """
    A class implementing the quasiperiodic kernel.

    :param array-like alpha: (4,) The parameter vector (amplitude,
    decay time, period, structure param).
    """
    def __init__(self, alpha):
        """
        :param array-like alpha: (4,) The parameter vector (amplitude,
        decay time, period, structure param).
        """
        super(QuasiPeriodicKernel, self).__init__(alpha)

    def _covariance(self, dx):
        alpha = self.hyperparams
        return alpha[0] ** 2 * np.exp(-0.5 * dx ** 2 / alpha[1] ** 2 - 2.0 *
                                      np.sin((np.pi * dx / alpha[2])) ** 2 /
                                      alpha[3] ** 2)


class DiagonalKernel(Kernel):
    """
    A convenience class to produce diagonal covariance functions.
    """
    def __init__(self, alpha=np.array([])):
        super(DiagonalKernel, self).__init__(alpha)

    def _covariance(self, dx):
        return np.eye(*dx.shape)
