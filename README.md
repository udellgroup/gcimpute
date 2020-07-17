# online_mixed_gc_imp
This package provides a python implemention of online, minibatch, and parallel implementations of Gaussian Copula Imputation.

## ExpectationMaximization
Implements an approximate expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism. An Example of usage can be found in evaluation/copula_generate/standard_copula_generated.py

## BatchExpectationMaximization
Implements an approximate minibatch expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches. An Example of usage can be found in evaluation/copula_generate/standard_copula_generated.py

## OnlineExpectationMaximization
Implements an approximate online expectation maximization algorithm for fitting a Guassian Copula to impute missing values that supports parallelism within minibatches. An Example of usage can be found in evaluation/copula_generate/standard_copula_generated.py