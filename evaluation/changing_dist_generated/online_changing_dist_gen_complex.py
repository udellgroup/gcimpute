from em.online_expectation_maximization import OnlineExpectationMaximization
from em.batch_expectation_maximization import BatchExpectationMaximization
import numpy as np
from evaluation.helpers import *
import matplotlib.pyplot as plt
import time
from scipy.stats import random_correlation, norm, expon
import pandas as pd

def main(START=1, NUM_RUNS=20):
    NUM_SAMPLES = 2000
    BATCH_SIZE = 40
    WINDOW_SIZE = 200
    NUM_ORD_UPDATES = 2
    NUM_BATCH = int(NUM_SAMPLES*3/BATCH_SIZE)
    smae_online_trials = np.zeros((NUM_RUNS, NUM_BATCH, 3))
    smae_offline_trials = np.zeros((NUM_RUNS, NUM_BATCH, 3))
    res_change_statistics = []
    for i in range(START, NUM_RUNS+START):
        print("starting epoch: ", i, "\n")
        sigma_window = []
        sigma_window_len = 1
        change_statistics = []
        
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
        
        # offline 
        bem = BatchExpectationMaximization() # Switch to batch implementation for acceleration
        start_time = time.time()
        X_imp_offline, _ = bem.impute_missing(X_masked, max_workers=4, batch_c=5, max_iter=2*NUM_BATCH)
        end_time = time.time()

        
        # online 
        oem = OnlineExpectationMaximization(cont_indices, ord_indices, window_size=WINDOW_SIZE)
        j = 0
        X_imp_online = np.zeros(X_masked.shape)
        #print(X_masked.shape, X_imp.shape, X.shape)
        Med = np.nanmedian(X_masked,0)
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
            # imputation error at each batch
            #smae_online_trials[i-1,j,:] = get_smae_per_type(X_imp_online[start:end,:], X[start:end,:], X_masked[start:end,:])
            #smae_offline_trials[i-1,j,:] = get_smae_per_type(X_imp_offline[start:end,:], X[start:end,:], X_masked[start:end,:])
            smae_online_trials[i-1,j,:] = get_smae(X_imp_online[start:end,:], X[start:end,:], X_masked[start:end,:], Med, True)
            smae_offline_trials[i-1,j,:] = get_smae(X_imp_offline[start:end,:], X[start:end,:], X_masked[start:end,:], Med, True)
            j += 1
        change_statistics = np.array(change_statistics)
        res_change_statistics.append(change_statistics)

    
    smae_online= np.mean(smae_online_trials, 0)
    smae_offline= np.mean(smae_offline_trials, 0)
    smae_means = pd.DataFrame(np.concatenate((smae_online, smae_offline), 1))
    smae_means.to_csv("sim_smae_means.csv")
    smae_online= np.std(smae_online_trials, 0)
    smae_offline= np.std(smae_offline_trials, 0)
    smae_stds = pd.DataFrame(np.concatenate((smae_online, smae_offline), 1))
    smae_stds.to_csv("sim_smae_stds.csv")
    
    #pd.DataFrame(change_statistics[:,2]).to_csv("sim_change_statistics_Frobenius.csv")
    #pd.DataFrame(change_statistics[:,1]).to_csv("sim_change_statistics_Nuclear.csv")
    #pd.DataFrame(change_statistics[:,0]).to_csv("sim_change_statistics_Spectral.csv")
    res_change_statistics = np.array(res_change_statistics)
    mean_change_statistics = np.mean(res_change_statistics, 0)
    pd.DataFrame(mean_change_statistics).to_csv("sim_change_statistics.csv")
    
    return smae_means, smae_stds, res_change_statistics


if __name__ == "__main__":
    smae_means, smae_stds, change_statistics = main(1,10)
    #mean_change_statistics = np.mean(change_statistics, 0)
    #plt.scatter(range(mean_change_statistics.shape[0]), mean_change_statistics[:,2], s=10,c='blue')
    
