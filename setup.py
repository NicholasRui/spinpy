from setuptools import setup

setup(name='spinpy',
      version='0.1',
      description='A package to perform optimal control on spin ensembles.',
      author='Nicholas Z Rui',
      author_email='nrui@berkeley.edu',
      license='CC BY',
      packages=['spinpy'],
      install_requires=['numpy','matplotlib','qutip'],
      zip_safe=False)
