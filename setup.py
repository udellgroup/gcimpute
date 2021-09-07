from setuptools import setup, find_packages

setup(
    name='GaussianCopulaImp',
    version='0.1',
    description='Online Expectation Maximization for Missing Value Imputation',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'tqdm'
    ],
)