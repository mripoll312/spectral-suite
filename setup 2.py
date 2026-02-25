#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation file for the data toolbox.

Author: Niels Krausch
"""

import os

from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='data_toolbox',
    version='0.11.2',
    author='Niels Krausch, Robert Giessmann',
    author_email='n.krausch@campus.tu-berlin.de, r.giessmann@tu-berlin.de',
    description='A data toolbox for analyzing kinetic reactions.',
    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',
    platforms='any',
    install_requires=[
        'lmfit',
        'numpy',
        'pandas>0.23.0',
        'matplotlib',
        'scipy',
        'configargparse',
    ],
    long_description=read('README.md'),
    entry_points={
        'console_scripts': [
            'data_toolbox=data_toolbox.data_toolbox:main',
        ],
    }
)
