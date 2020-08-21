import numpy as np
from em.expectation_maximization import ExpectationMaximization
from scipy.stats import random_correlation, norm, expon
from evaluation.helpers import *
import pandas as pd
import time

def generate_sigma(seed=0):
    np.random.seed(seed)
    W = np.random.normal(size=(15,15))
    covariance = np.matmul(W,W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

if __name__ == "__main__":
    scaled_errors = []
    smaes = []
    rmses = []
    NUM_STEPS = 3
    runtimes = []
    for i in range(1, NUM_STEPS + 1):
        print("starting epoch: " + str(i))
        print("\n")
        sigma = generate_sigma(i)
        mean = np.zeros(sigma.shape[0])
        np.random.seed(i)
        X = np.random.multivariate_normal(mean, sigma, size=2000)
        X[:,:5] = expon.ppf(norm.cdf(X[:,:5]), scale = 3)
        for j in range(5,15,1):
            # 6-10 columns are binary, 11-15 columns are ordinal with 5 levels
            X[:,j] = cont_to_ord(X[:,j], k=2*(j<10)+5*(j>=10))
        # mask a given % of entries
        MASK_FRACTION = 0.3
        X_masked, mask_indices = mask(X, MASK_FRACTION, seed=i)
        em = ExpectationMaximization()
        start_time = time.time()
        X_imp, sigma_imp = em.impute_missing(X_masked, threshold=0.01)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        rmse = get_scaled_error(X_imp[:,:5], X[:,:5])
        scaled_error = get_scaled_error(sigma_imp, sigma)
        smae = get_smae(X_imp, X, X_masked)
        # update error to be normalized
        scaled_errors.append(scaled_error)
        smaes.append(smae)
        rmses.append(rmse)
    print("mean of scaled errors is: ")
    print(np.mean(np.array(scaled_errors)))
    print("std deviation of scaled errors is: ")
    print(np.std(np.array(scaled_errors)))
    print("\n")
    mean_smaes = np.mean(np.array(smaes),axis=0)
    print("mean cont smaes are: ")
    print(np.mean(mean_smaes[:5]))
    print("mean bin smaes are: ")
    print(np.mean(mean_smaes[5:10]))
    print("mean ord smaes are: ")
    print(np.mean(mean_smaes[10:]))
    print("\n")
    std_dev_smaes = np.std(np.array(smaes),axis=0)
    print("std dev cont smaes are: ")
    print(np.mean(std_dev_smaes[:5]))
    print("std dev bin smaes are: ")
    print(np.mean(std_dev_smaes[5:10]))
    print("std dev ord smaes are: ")
    print(np.mean(std_dev_smaes[10:]))
    print("\n")
    print("mean of rmses is: ")
    print(np.mean(np.array(rmses),axis=0))
    print("std deviation of rmses is: ")
    print(np.std(np.array(rmses),axis=0))
    print("\n")
    print("mean time for run is: ")
    print(np.mean(np.array(runtimes)))





