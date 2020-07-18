from setuptools import setup, find_packages

setup(
    name='online_gc_imp',
    version='0.1',
    description='Online Expectation Maximization for Missing Value Imputation',
    packages=['em', 'evaluation', 'transforms'],
    install_requires=[
        'numpy',
        'scyipy',
        'concurrent',
        'statsmodels'
    ],
)