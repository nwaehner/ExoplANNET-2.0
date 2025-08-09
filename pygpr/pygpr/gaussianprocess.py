"""
This module contains the Gaussian Process class.
class GaussianProcess: A Class implementing Gaussian processes.
"""
import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.linalg import cho_factor, cho_solve


class GaussianProcess(object):
    """
    A Class implementing Gaussian processes.

    Instances are constructed by providing a Kernel instance, an array of
    input test coordinates where the GP is defined, and optionally an array
    representing the data used to produce predictions.
    """
    def __init__(self, kernel, xinput, data=None):
        """
        :param kernel: an instance of the :class:`~gp.kernels.Kernel`
        :param np.array xinput: "test" input coordinates.
        :param np.array data: a `(N x 2)` or `(N x 3)` array of N data inputs:
         (data coordiante, data value, data error (optional)).
        """
        # Initialise input attributes (for PEP-8 compliance).
        self._input = None
        self._data = None
        self.covariance = None
        self.covariance_data = None
        self.covariance_test_data = None

        # Set kernel
        self.kernel = kernel

        # Set the input test coordinates
        self.x = xinput

        # Set data (if given).
        self.data = data

        # Initialize posterior mean and covariance to prior values
        self.predmean = np.zeros_like(xinput)
        self.predcov = self.covariance

    @property
    def x(self):
        """The GP test input coordinate vector."""
        return self._input

    @x.setter
    def x(self, inputarray):
        self._input = inputarray

        # Compute the diffrence matrix and covariance
        dx = self._input[:, None] - self._input[None, :]
        self.covariance = self.kernel.covariance(dx)

        if self.data is not None:
            cov_star_data, cov_data = self.computecovariances(self._data)
            self.covariance_test_data = cov_star_data

    @x.deleter
    def x(self):
        self._input = None
        self.covariance = None

    def get_test_input(self):
        """Return the array of input coordinates."""
        return self.x

    def set_test_input(self, inputarray):
        """Define inputarray as the GP input coordinates."""
        self.x = inputarray

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, dataarray):
        if dataarray is not None:
            self._data = dataarray
            cov_star_data, cov_data = self.computecovariances(self._data)
            self.covariance_test_data = cov_star_data
            self.covariance_data = cov_data

    @data.deleter
    def data(self):
        self.data = None
        self.covariance_data = None
        self.covariance_test_data = None

    def erasedata(self):
        """Erases the GP data array and resets the relevant covariances
        matrices."""
        del self.data

    def computecovariances(self, data):
        """
        Compute the covariances between the data inputs (data) and the test
        inputs (star).

        :param np.array data: a 2-D array with dimensions (2, n) or (3, n).
        :returns: two covariances matrices
        """
        xdata = data[0]
        dx_star_data = self.x[:, None] - xdata[None, :]
        dx_data = xdata[:, None] - xdata[None, :]
        return self.kernel.covariance(dx_star_data), self.kernel.covariance(
            dx_data)

    def sample(self, size=1):
        """
        Produce a sample from the GP functions.
        :param int size: the size of the sample.
        :return np.array: a (s, n) array, with s the sample size and n the
        length of the test input array.
        """
        return np.random.multivariate_normal(np.zeros_like(self.x),
                                             self.covariance, size)

    def prediction(self, data=None):
        """
        Evaluates the posterior GP mean and covariance functions.

        This method computes the mean and covariance matrix of the posterior
        predictive distribution of the GP. The mean and covariance matrix are
        incorporated as attributes of the class and can be subsequently used to
        draw samples of the function values corresponding to the input values.

        If no data array is passed as argument, then the data attribute is used.

        :param np.array data: a `(N x 2)` or `(N x 3)` array of N data inputs:
         (data coordiante, data value, data error (optional)).

        :return: mean and covariance matrix of posterior predictive.
        """

        if data is None and self.data is None:
            raise TypeError('Data array cannot be None, unless you want your'
                            'predictions to look like your prior. In that'
                            'case, better use the `sample` method.')

        elif data is not None:

            if self.data is not None:
                print('Data given. Overriden previous data.')
            self.data = data

            # Compute covariance matrices
            cov_test_data, cov_data = self.computecovariances(self.data)
            self.covariance_test_data = cov_test_data
            self.covariance_data = cov_data

        # If errors are provided for data, add them to the covariance diagonal
        if self.data.shape[0] > 2:
            dataerror = np.diag(np.atleast_1d(self.data[2] ** 2))
        else:
            dataerror = np.diag(np.zeros_like(self.data[0]))

        # Use Cholesky decomposition on covariance of data inputs.
        factor, flag = cho_factor(self.covariance_data + dataerror)

        # Compute posterior mean (eq. 2.23 Rasmussen)
        a = cho_solve((factor, flag), self.data[1])
        self.predmean = np.dot(self.covariance_test_data, np.array(a))

        # Compute posterior covariance (eq. 2.24 Rasmussen)
        alpha = cho_solve((factor, flag), self.covariance_test_data.T)
        beta = np.dot(self.covariance_test_data, np.array(alpha))
        self.predcov = self.covariance - beta

        return self.predmean, self.predcov

    def prediction_sample(self, size=1):
        """
        Sample function values from the GP prediction.

        :param int size: sample size to draw
        :return np.array: a (s, n) array, with s the sample size and n the
                          length of the test input array.
        """
        if np.array_equal(self.predcov, self.covariance):
            raise RuntimeWarning('Posterior covariance is identical to prior '
                                 'covariance. Try using the prediction method '
                                 'first.')
        return mvn(mean=self.predmean, cov=self.predcov, size=size)