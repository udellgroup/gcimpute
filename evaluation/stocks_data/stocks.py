from em.online_expectation_maximization import OnlineExpectationMaximization
from em.batch_expectation_maximization import BatchExpectationMaximization
import numpy as np
from evaluation.helpers import *
import matplotlib.pyplot as plt
import time
from scipy.stats import random_correlation, norm, expon
import pandas as pd


def main(START=1, NUM_RUNS=10,seed_last=0):
    #data_iq = np.array(pd.read_csv('dengue_iq.csv'))[:,1:]
    #X = data_iq
    X= np.array(pd.read_csv('/Users/yuxuan/Desktop/Stock_data/log_return_DJIA.csv'))[:,2:]
    X = X.astype(np.float)
    n,p = X.shape

    
    MASK_FRACTION = 0.4
    WINDOW_SIZE = 25 # bug at 25, if set as 50 everthing works fine
    # It should work for any window size.
    decay_coef = 0.5
    BATCH_SIZE = 40
    NUM_BATCH = int(n/BATCH_SIZE) + 1
    
    smae_online_trials = np.zeros((NUM_RUNS, NUM_BATCH))
    smae_offline_trials = np.zeros((NUM_RUNS, NUM_BATCH))
    for i in range(START, NUM_RUNS+START):
        print("starting epoch: ", i, "\n")

        cont_indices = np.array([True] * p)
        ord_indices = np.array([False] * p)
        
        X_masked, mask_indices, seed_last = mask(X, MASK_FRACTION, seed=seed_last+1)
        
        # offline 
        bem = BatchExpectationMaximization() # Switch to batch implementation for acceleration
        start_time = time.time()
        X_imp_offline, _ = bem.impute_missing(X_masked, max_workers=4, batch_c=5, max_iter=2*NUM_BATCH)
        end_time = time.time()
        print("offline: "+str(time.time() - start_time))
        
        # online 
        oem = OnlineExpectationMaximization(cont_indices, ord_indices, window_size=WINDOW_SIZE)
        j = 0
        X_imp_online = np.zeros(X_masked.shape)
        #print(X_masked.shape, X_imp.shape, X.shape)
        Med = np.nanmedian(X_masked,0)
        start_time = time.time()
        while True:
            start = j*BATCH_SIZE
            if start >= n:
            	break
            end = (j+1)*BATCH_SIZE
            X_masked_batch = np.copy(X_masked[start:end,:])
            X_imp_online[start:end,:] = oem.partial_fit_and_predict(X_masked_batch, 
                                                             max_workers = 4, 
                                                             decay_coef=decay_coef)

            # imputation error at each batch
            #smae_online_trials[i-1,j,:] = get_smae_per_type(X_imp_online[start:end,:], X[start:end,:], X_masked[start:end,:])
            #smae_offline_trials[i-1,j,:] = get_smae_per_type(X_imp_offline[start:end,:], X[start:end,:], X_masked[start:end,:])
            smae_online_trials[i-1,j] = np.nanmean(get_smae(X_imp_online[start:end,:], X[start:end,:], X_masked[start:end,:], Med))
            smae_offline_trials[i-1,j] = np.nanmean(get_smae(X_imp_offline[start:end,:], X[start:end,:], X_masked[start:end,:], Med))
            j += 1
        end_time = time.time()
        print("online: "+str(time.time() - start_time))



    #pd.DataFrame(smae_online_trials).to_csv("stocks_online_smae.csv")
    #pd.DataFrame(smae_offline_trials).to_csv("stocks_offline_smae.csv")

    return smae_online_trials, smae_offline_trials
  


if __name__ == "__main__":
    online, offline = main(1,10)
