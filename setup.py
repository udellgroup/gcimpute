from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='GaussianCopulaImp',
    version='0.0.6',
    description='A missing value imputation algorithm using the Gaussian copula model',
    long_description=long_description,
    long_description_content_type='text/markdown',
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