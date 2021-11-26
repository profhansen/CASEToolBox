from setuptools import setup, find_packages

setup(name='casetoolbox',
      version='0.0.1',
      description='Computational aero-servo-elastic toolbox',
      url='',
      author='Morten Hartvig Hansen',
      author_email='morten.hartvig.hansen@gmail.com',
      license='GNU AGPLv3',
      packages=find_packages(),
      py_packages=['casestab'],
      install_requires=[
          'numpy','numba','scipy','matplotlib',
      ],
      zip_safe=False)