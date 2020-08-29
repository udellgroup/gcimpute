import numpy as np
from em.online_expectation_maximization import OnlineExpectationMaximization
from scipy.stats import random_correlation, norm, expon
from evaluation.helpers import *
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
    runtimes = []
    NUM_RUNS = 1
    n = 2000
    max_iter = 200
    for i in range(1,NUM_RUNS+1):
        np.random.seed(i)
        print("starting epoch: " + str(i))
        print("\n")
        sigma = generate_sigma(seed=i)
        mean = np.zeros(sigma.shape[0])
        X = np.random.multivariate_normal(mean, sigma, size=n)
        #X[:,:5] = expon.ppf(norm.cdf(X[:,:5]), scale = 3)
        #for j in range(5,15,1):
            # 6-10 columns are binary, 11-15 columns are ordinal with 5 levels
         #   X[:,j] = cont_to_ord(X[:,j], k=2*(j<10)+5*(j>=10))
        #cont_indices = np.array([True] * 5 + [False] * 10)
        #ord_indices = np.array([False] * 5 + [True] * 10)
        for j in range(10):
            X[:,j] = cont_to_ord(X[:,j], k=2*(j<5)+5*(j>=5))
        X[:,10:] = expon.ppf(norm.cdf(X[:,10:]), scale = 3)
        cont_indices = np.array([False] * 10 + [True] * 5)
        ord_indices =  np.array([True] * 10 + [False] * 5)
        # mask a given % of entries
        MASK_NUM = 2
        X_masked, mask_indices = mask_types(X, MASK_NUM, seed=i)
        BATCH_SIZE=40
        WINDOW_SIZE=500
        
        oem = OnlineExpectationMaximization(cont_indices, ord_indices, window_size=WINDOW_SIZE)
        start_time = time.time()
        j = 0
        X_imp = np.empty(X_masked.shape)
        scaled_errors = []
        while j <= max_iter:  
            start = (j*BATCH_SIZE) % n
            end = ((j+1)*BATCH_SIZE) % n
            if end < start:
                indices = np.concatenate((np.arange(end), np.arange(start, n, 1)))
            else:
                indices = np.arange(start, end, 1)
            X_imp[indices,:] = oem.partial_fit_and_predict(X_masked[indices,:],max_workers = 4, decay_coef=0.5)
            j +=1 
        end_time = time.time()
        runtimes.append(end_time - start_time)
        # correlation estimation
        sigma_imp = oem.get_sigma()
        scaled_error = get_scaled_error(sigma_imp, sigma)
        scaled_errors.append(scaled_error)
        # imputation 
        smae = get_smae(X_imp, X, X_masked)
        smaes.append(smae)
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
    print("mean time for run is: ")
    print(np.mean(np.array(runtimes)))