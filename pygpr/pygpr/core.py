import numpy as np
import kernels
from gaussianprocess import GaussianProcess


def sample_gp(x, gp=None, alpha=np.empty(0), kerneltype='se', size=1):
    """
    :param array x: input array where GP is to be sampled.
    :param gp: a `~gp.core.GaussianProcess` class instance. If None,
    the instance will be constructed using hyperparameter vector alpha and
    kerneltype.
    :param array-like alpha: List of hyperparameters to build kernel instance.
    :param string kerneltype: The kernel type. Options are:
        'se' for a squared exponential kernel.
        'qper' for a quasiperiodic kernel.
    :param int size: size of the sample.
    """
    # If a Kernel instance is passed.
    if isinstance(gp, GaussianProcess):
        gp.set_test_input(x)
        return gp.sample(size)
    # Otherwise, construct instance on the fly.
    elif kerneltype == 'se':
        return kernels.SquaredExponentialKernel(alpha).sample(x, size)
    elif kerneltype == 'ge':
        return kernels.GeneralisedExponentialKernel(alpha).sample(x, size)
    elif kerneltype == 'qper':
        return kernels.QuasiPeriodicKernel(alpha).sample(x, size)
    else:
        raise NameError('Kerneltype not recognised.')
