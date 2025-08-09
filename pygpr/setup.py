from distutils.core import setup

setup(name='pygpr',
      description='Basic Gaussian process regression tasks in pure python.',
      version='0.1.0',
      author='Rodrigo F. Diaz',
      author_email='rodrigo.diaz@unige.ch',
      url='http://obswww.unige.ch/~diazr/pygpr',
      packages=['pygpr'],
      requires=['numpy', 'scipy.linalg']
      )
