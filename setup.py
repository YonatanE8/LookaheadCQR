from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(name='LookaheadCQR',
      version='1.0',
      description=(
          'This Package Implements the CQR + Lookahead regularization final project for'
          ' the robustness and reliability in deep-learning course.'
      ),
      author='Yonatan Elul, Moshe Kimhi',
      author_email='johnneye@campus.technion.ac.il, moshekimhi@campus.technion.ac.il',
      url='https://github.com/YonatanE8/LookaheadCQR.git',
      license='',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Private',
          'Topic :: Software Development :: Deep Learning',
          'Programming Language :: Python :: 3.8',
      ],
      package_dir={'LookaheadCQR': os.path.join(cwd, 'LookaheadCQR')},
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib',
          'torch',
          'torchvision',
          'tqdm',
          'pandas',
          'pygam',
          'cycler',
          'jupyter',
          #'scikit-garden',
      ],
      )
