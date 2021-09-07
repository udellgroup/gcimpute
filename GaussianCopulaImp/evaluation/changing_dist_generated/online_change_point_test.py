from em.online_expectation_maximization import OnlineExpectationMaximization
from em.batch_expectation_maximization import BatchExpectationMaximization
import numpy as np
from evaluation.helpers import *
import matplotlib.pyplot as plt
import time
from scipy.stats import random_correlation, norm, expon
import pandas as pd


def main(START=1, NUM_RUNS=10):
    NUM_SAMPLES = 2000
    BATCH_SIZE = 40
    WINDOW_SIZE = 200
    NUM_BATCH = int(NUM_SAMPLES*3/BATCH_SIZE)
    
    
    for i in range(START, NUM_RUNS+START):
        q = []
        s = []
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
        j = 0
        start_time = time.time()
        while j<NUM_BATCH:
            start = j*BATCH_SIZE
            end = (j+1)*BATCH_SIZE
            x_batch = X_masked[start:end,:]
            if j== 0:
                oem.partial_fit_and_predict(x_batch, max_workers = 4, decay_coef=0.5)
            else:
                pval_iter, s_iter = oem.change_point_test(x_batch, decay_coef=0.5, nsample=200, max_workers=4)
                print(s_iter)
                print(pval_iter)
                q.append(pval_iter)
                s.append(s_iter)
            # oem.partial_fit_and_predict(X[start:end,:], max_workers = 4, decay_coef=decay_coef)
            print("finish epoch: ", j, "\n")
            j += 1
        print("online: "+str(time.time() - start_time))
        pd.DataFrame(np.array(q)).to_csv("sim_change_pvalues_rep_"+str(i)+".csv")
        pd.DataFrame(np.array(s)).to_csv("sim_change_statistics_rep_"+str(i)+".csv")
        print("finish iteration "+str(i))

    
    

if __name__ == "__main__":
    main(1,10)