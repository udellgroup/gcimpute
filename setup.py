from setuptools import setup, find_packages

setup(
    name='GaussianCopulaImp',
    version='0.0.5',
    description='A missing value imputation algorithm using the Gaussian copula model',
    packages=find_packages(),
    url='https://github.com/udellgroup/GaussianCopulaImp',
    author = 'Yuxuan Zhao, Eric Landgrebe, Eliot Shekhtman, Madeleiene Udell',
    author_email = 'yz2295@cornell.edu',
    maintainer = 'Yuxuan Zhao',
    maintainer_email = 'yz2295@cornell.edu',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'tqdm'
    ],
    include_package_data=True,
    package_data = {'':['data/*.csv']},
)