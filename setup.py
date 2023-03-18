# -*- coding: utf-8 -*-

# Learn more: https://github.com/chandrakar-shubham/bike_sharing_demand_pred.git

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='bike_sharing_demand_prediction',
    version='1.0.0',
    description='Sample package for Python-Guide.org',
    long_description=readme,
    author='Shubham Chandrakar',
    author_email='chandrakar.shubham17@gmail.com',
    url='https://github.com/chandrakar-shubham/bike_sharing_demand_pred',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
    install_requires=requirements)

