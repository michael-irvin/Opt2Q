# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='Opt2Q',
      version='0.3.1',
      description='Calibrating systems biology dynamical models with quantitative and non-quantitative measurements.',
      long_description=long_description,
      url='https://github.com/michael-irvin/Opt2Q',
      author='Michael Irvin',
      author_email='michael.w.irvin@vanderbilt.edu',
      license='MIT',

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=['Development Status :: 3 - Alpha',],
      keywords=['systems biology', 'dynamics', 'biological pathways',
                'non-quantitative', 'measurements'],

      # You can just specify the packages manually here if your project is
      # simple. Or you can use find_packages().
      packages=find_packages(exclude=['docs']),
      install_requires=['nose',
                        'numpy',
                        'pandas',
                        'cython',
                        'scipy',
                        'matplotlib'],
      )
