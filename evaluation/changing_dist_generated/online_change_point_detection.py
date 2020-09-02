from em.online_expectation_maximization import OnlineExpectationMaximization
from em.batch_expectation_maximization import BatchExpectationMaximization
import numpy as np
from evaluation.helpers import *
import matplotlib.pyplot as plt
import time
from scipy.stats import random_correlation, norm, expon
import pandas as pd
import matplotlib.pyplot as plt

def main(START=1, NUM_RUNS=10):
    NUM_SAMPLES = 10000
    BATCH_SIZE = 40
    WINDOW_SIZE = 200
    NUM_ORD_UPDATES = 2
    NUM_BATCH = int(NUM_SAMPLES*3/BATCH_SIZE)

    sigma_window = []
    sigma_window_len = 2
    change_statistics = []
    for i in range(START, NUM_RUNS+START):
        print("starting epoch: ", i, "\n")
        sigma1 = generate_sigma(3*i-2)
        sigma2 = generate_sigma(3*i-1)
        sigma3 = generate_sigma(3*i)
        mean = np.zeros(sigma1.shape[0])
        X1 = np.random.multivariate_normal(mean, sigma1, size=NUM_SAMPLES)
        X2 = np.random.multivariate_normal(mean, sigma2, size=NUM_SAMPLES)
        X3 = np.random.multivariate_normal(mean, sigma3, size=NUM_SAMPLES)
        X = np.vstack((X1, X2, X3))
        X = np.vstack((X1, X2, X3))
        X[:,:5] = expon.ppf(norm.cdf(X[:,:5]), scale = 3)
        for j in range(5,15,1):
            # 6-10 columns are binary, 11-15 columns are ordinal with 5 levels
            X[:,j] = cont_to_ord(X[:,j], k=2*(j<10)+5*(j>=10))
        cont_indices = np.array([True] * 5 + [False] * 10)
        ord_indices = np.array([False] * 5 + [True] * 10)

        # X_masked = mask_one_per_row(X)
        MASK_NUM = 2
        X_masked, mask_indices = mask_types(X, MASK_NUM, seed=i)

        
        # online 
        oem = OnlineExpectationMaximization(cont_indices, ord_indices, window_size=WINDOW_SIZE)
        X_imp = np.empty(X.shape)
        j = 0
        X_imp_online = np.zeros(X_masked.shape)
        while j<NUM_BATCH:
            start = j*BATCH_SIZE
            end = (j+1)*BATCH_SIZE
            X_masked_batch = np.copy(X_masked[start:end,:])
            X_imp_online[start:end,:] = oem.partial_fit_and_predict(X_masked_batch, 
                                                             max_workers = 4, 
                                                             decay_coef=0.5, 
                                                             num_ord_updates=NUM_ORD_UPDATES)
            if j>=sigma_window_len:
                sigma_new = oem.get_sigma()
                dist = np.zeros((sigma_window_len, 3))
                for t,sigma_old in enumerate(sigma_window):
                    u, s, vh = np.linalg.svd(sigma_old)
                    factor = (u * np.sqrt(1/s) ) @ vh
                    diff = factor @ sigma_new @ factor
                    _, s, _ = np.linalg.svd(diff)
                    dist[t,:] = max(abs(s-1)), np.sum(abs(s-1)), np.linalg.norm(diff-np.identity(15))
                change_statistics.append(np.max(dist,0))
                sigma_window.pop(0)
                sigma_window.append(sigma_new)
            else:
                sigma_window.append(oem.get_sigma())
            j += 1
        change_statistics = np.array(change_statistics)
        return change_statistics
    

if __name__ == "__main__":
    change_statistics = main(1,10)
    
    #plt.scatter(range(change_statistics.shape[0]), change_statistics[:,0], s=10,c='blue')
    #plt.scatter(range(change_statistics.shape[0]), change_statistics[:,1], s=10,c='blue')
    #plt.scatter(range(change_statistics.shape[0]), change_statistics[:,2], s=10,c='blue')