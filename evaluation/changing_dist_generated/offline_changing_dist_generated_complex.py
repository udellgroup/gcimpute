from em.expectation_maximization import ExpectationMaximization
import numpy as np
from evaluation.helpers import cont_to_binary, cont_to_ord, get_smae, get_rmse, get_scaled_error, mask, mask_one_per_row
from transforms.online_ordinal_marginal_estimator import OnlineOrdinalMarginalEstimator
from statsmodels.distributions.empirical_distribution import ECDF
import time
from scipy.stats import random_correlation, norm, expon
import pandas as pd


def generate_sigma(seed):
    np.random.seed(seed)
    W = np.random.normal(size=(15, 15))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def get_smae_types(X_imp, X, X_masked):
    smae_cont = get_smae(X_imp[:, :5], X[:, :5], X_masked[:, :5])
    smae_ord = get_smae(X_imp[:, 5:10], X[:, 5:10], X_masked[:, 5:10])
    smae_bin = get_smae(X_imp[:, 10:], X[:, 10:], X_masked[:, 10:])
    return (smae_cont, smae_ord, smae_bin)

def mask_types(X, mask_num, seed):
    """
    Masks mask_num entries of the continuous, ordinal, and binary columns of X
    """
    X_masked = np.copy(X)
    mask_indices = []
    for i in range(X_masked.shape[0]):
        np.random.seed(seed*X_masked.shape[0]-i) # uncertain if this is necessary
        rand_idx = np.concatenate((np.random.choice(5, mask_num, False), np.random.choice(5, mask_num, False), np.random.choice(5, mask_num, False)))
        for idx in rand_idx[:mask_num]:
            X_masked[i, idx] = np.nan
            mask_indices.append((i,idx))
        for idx in rand_idx[mask_num:2*mask_num]:
            X_masked[i, idx+5] = np.nan
            mask_indices.append((i,idx+5))
        for idx in rand_idx[2*mask_num:]:
            X_masked[i, idx+10] = np.nan
            mask_indices.append((i,idx+10))
    return X_masked, mask_indices

def avg_trials(data):
    num_trials = len(data)
    avgd_data = np.zeros(np.array(data[0]).shape)
    for trial in data:
        avgd_data += np.array(trial)
    return avgd_data / num_trials

if __name__ == "__main__":
    # scaled_errors = []
    smaes = []
    # rmses = []
    runtimes = []
    NUM_RUNS = 10
    NUM_SAMPLES = 2000
    BATCH_SIZE = 50
    smae_cont_trials = []
    smae_ord_trials = []
    smae_bin_trials = []
    for i in range(1, NUM_RUNS+1):
        smae_conts = []
        smae_ords = []
        smae_bins = []
        print("starting epoch: ", i, "\n")
        # scaled_errors = []
        sigma1 = generate_sigma(3*i-2)
        sigma2 = generate_sigma(3*i-1)
        sigma3 = generate_sigma(3*i)
        mean = np.zeros(sigma1.shape[0])
        X1 = np.random.multivariate_normal(mean, sigma1, size=NUM_SAMPLES)
        X2 = np.random.multivariate_normal(mean, sigma2, size=NUM_SAMPLES)
        X3 = np.random.multivariate_normal(mean, sigma3, size=NUM_SAMPLES)
        X = np.vstack((X1, X2, X3))
        # X[:, 1] = cont_to_ord(X[:, 1], k=5)
        # X[:, 2] = cont_to_binary(X[:, 2])

        X[:, :5] = expon.ppf(norm.cdf(X[:, :5]), scale=3)
        X[:, 5] = cont_to_binary(X[:, 5])
        X[:, 6] = cont_to_binary(X[:, 6])
        X[:, 7] = cont_to_binary(X[:, 7])
        X[:, 8] = cont_to_binary(X[:, 8])
        X[:, 9] = cont_to_binary(X[:, 9])
        X[:, 10] = cont_to_ord(X[:, 10], k=5)
        X[:, 11] = cont_to_ord(X[:, 11], k=5)
        X[:, 12] = cont_to_ord(X[:, 12], k=5)
        X[:, 13] = cont_to_ord(X[:, 13], k=5)
        X[:, 14] = cont_to_ord(X[:, 14], k=5)

        # X_masked = mask_one_per_row(X)
        MASK_NUM = 2
        X_masked, mask_indices = mask_types(X, MASK_NUM, seed=i)
        print(X_masked[0])
        cont_indices = np.array([True, True, True, True, True, False,
                                 False, False, False, False, False, False, False, False, False])
        ord_indices = np.array([False, False, False, False, False,
                                True, True, True, True, True, True, True, True, True, True])
        em = ExpectationMaximization()
        start_time = time.time()
        X_imp, sigma_imp = em.impute_missing(X_masked, threshold=0.01)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        smae_conts, smae_ords, smae_bins = get_smae_types(X_imp, X, X_masked)
        # smae_cont, smae_ord, smae_bin = get_smae_types(X_imp, X, X_masked)
        smae_cont_trials.append(smae_conts)
        smae_ord_trials.append(smae_ords)
        smae_bin_trials.append(smae_bins)
    mean_smae = np.mean(smaes, axis=0)

    for i in range(NUM_RUNS):
        d = {'Continuous': smae_cont_trials[i], 'Ordinal': smae_ord_trials[i], 'Binary': smae_bin_trials[i]}
        df = pd.DataFrame(d)
        df.to_csv('data_'+str(i)+'_std.csv')
    smae_cont_trials = avg_trials(smae_cont_trials)
    smae_ord_trials = avg_trials(smae_ord_trials)
    smae_bin_trials = avg_trials(smae_bin_trials)
    mean_smaes = np.mean(np.array(smaes), axis=0)
    print("mean cont smaes are: ")
    print(np.mean(mean_smaes[:5]))
    print("mean bin smaes are: ")
    print(np.mean(mean_smaes[5:10]))
    print("mean ord smaes are: ")
    print(np.mean(mean_smaes[10:]))
    print("\n")
    std_dev_smaes = np.std(np.array(smaes), axis=0)
    print("std dev cont smaes are: ")
    print(np.mean(std_dev_smaes[:5]))
    print("std dev bin smaes are: ")
    print(np.mean(std_dev_smaes[5:10]))
    print("std dev ord smaes are: ")
    print(np.mean(std_dev_smaes[10:]))
    print("\n")
    # print("mean of rmses is: ")
    # print(np.mean(np.array(rmses), axis=0))
    # print("std deviation of rmses is: ")
    # print(np.std(np.array(rmses), axis=0))
    # print("\n")
    print("mean time for run is: ")
    print(np.mean(np.array(runtimes)))
