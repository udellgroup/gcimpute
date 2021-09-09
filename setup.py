from setuptools import setup, find_packages

setup(
    name='GaussianCopulaImp',
    version='0.1',
    description='Online Expectation Maximization for Missing Value Imputation',
    packages=find_packages(),
    url='https://github.com/udellgroup/online_mixed_gc_imp',
    author = 'Eric Landgrebe, Yuxuan Zhao, Eliot Shekhtman',
    author_email = '{ecl93, yz2295, ess239}@cornell.edu',
    maintainer = 'Yuxuan Zhao',
    maintainer_email = 'yz2295@cornell.edu',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'tqdm'
    ]
)