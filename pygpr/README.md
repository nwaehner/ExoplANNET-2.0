.. gp documentation master file, created by
   sphinx-quickstart on Tue Apr 26 15:54:25 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygpr
=====

Description
-----------
``pygpr`` is a python package to perform basic Gaussian Process (GP) regression tasks, such as sampling function values from a GP, or evaluating the posterior predictive distribution given some data.

See :ref:`detailed-description` for links to the python classes.

Usage example
-------------

A short example using a squared exponential kernel. You can copy the code blocs below and paste them into a Ipython terminal using the magic function %paste::

    import numpy as np
    import pylab as plt
    plt.ion()

    import pygpr
   
    # Create an instance of a squared exponential kernel.
    # Hyper parameters are: 1.0 for the amplitude and 0.1 for the length scale.
    sek = pygpr.kernels.SquaredExponentialKernel([1.0, 0.1])

    # Define the sample input points where the GP will be defined.
    x = np.linspace(0.0, 1.0, 400)

    # Build a GP instance passing the Kernel instance and the input array.
    mygp = pygpr.GaussianProcess(sek, x)

    # Sample the GP function values
    f = mygp.sample(size=5)

    # We can now plot the five samples.
    plt.plot(x, f.T)


Interesting things start to happen when we add some data to the story::

    # Create a data array with two (noiseless) datapoints.
    data = np.array([[0.2, 0.65], [-1.5, 1.]])
    mygp.data = data

    # Given these data, we can find the 
    # GP prediction for the function values.
    mean, cov = mygp.prediction()

    # And we can obtain samples from the predicted functions
    fp = mygp.prediction_sample(size=20)
    fig = plt.figure()
    plt.plot(x, fp.T)

    # Add the datapoints
    plt.plot(data[0], data[1], 'or', ms=10)

We can add a further datapoint and see how the prediction changes::

    data2 = np.array([[0.2, 0.5, 0.65], [-1.5, -1.2, 1.]])

    mean, cov = mygp.prediction(data=data2)

    fp = mygp.prediction_sample(size=20)
    fig = plt.figure()
    plt.plot(x, fp.T)

    # Add the datapoints
    plt.plot(mygp.data[0], mygp.data[1], 'or', ms=10)

A more detailed example will be included in the :ref:`tutorial`.

.. _detailed-description:

Detailed description
--------------------
The package implements the GaussianProcess :ref:py:class:gp: GaussianProcess class and the 

Contents:

.. toctree::
   :maxdepth: 3

   rstfiles/installation
   rstfiles/tutorial
   rstfiles/api
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

