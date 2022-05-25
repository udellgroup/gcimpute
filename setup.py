from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='gcimpute',
    version='0.0.3',
    description='A missing value imputation algorithm using the Gaussian copula model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/udellgroup/gcimpute',
    author = 'Yuxuan Zhao, Eric Landgrebe, Eliot Shekhtman, Madeleiene Udell',
    author_email = 'yz2295@cornell.edu',
    maintainer = 'Yuxuan Zhao',
    maintainer_email = 'yz2295@cornell.edu',
    install_requires=[
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'scipy>=1.4.1',
        'statsmodels>=0.11.0',
        'tqdm>=4.42.1',
        'importlib_resources'
    ],
    include_package_data=True,
    package_data = {'gcimpute':['data/*.csv']},
    license = 'MIT',
    python_requires=">=3.7"
)