#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension

setup(name='dynamic_mos',
      packages=['dynamic_mos', 'adversarial_mos'],
      version='0.0',
      description='Dynamic multi-object search',
      python_requires='>3.6',
      install_requires=[
          'pyyaml',
          'numpy',
          'matplotlib',
          'networkx',
          'pygraphviz',
          'pomdp_py'
      ],
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com'
     )

