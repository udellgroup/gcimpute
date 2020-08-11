from em.online_expectation_maximization import OnlineExpectationMaximization
import numpy as np
from evaluation.helpers import cont_to_binary, cont_to_ord, get_smae, get_rmse, get_scaled_error, mask, mask_one_per_row
from transforms.online_ordinal_marginal_estimator import OnlineOrdinalMarginalEstimator
from statsmodels.distributions.empirical_distribution import ECDF
import time
from scipy.stats import random_correlation, norm, expon


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

if __name__ == "__main__":
    # scaled_errors = []
    smaes = []
    rmses = []
    runtimes = []
    NUM_RUNS = 10
    NUM_SAMPLES = 2000
    BATCH_SIZE = 50
    for i in range(1, NUM_RUNS+1):
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
        MASK_FRACTION = 0.3
        X_masked, mask_indices = mask(X, MASK_FRACTION)
        cont_indices = np.array([True, True, True, True, True, False,
                                 False, False, False, False, False, False, False, False, False])
        ord_indices = np.array([False, False, False, False, False,
                                True, True, True, True, True, True, True, True, True, True])
        oem = OnlineExpectationMaximization(cont_indices, ord_indices)
        X_imp = np.empty(X.shape)
        i = 0
        start_time = time.time()
        while i*BATCH_SIZE < X_masked.shape[0]:
            X_imp[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], sigma_imp = oem.partial_fit_and_predict(
                X_masked[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :])
            # smae = get_smae(X_imp[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], X[i*BATCH_SIZE:(
            #     i+1)*BATCH_SIZE, :], X_masked[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :])
            i += 1
            # scaled_errors.append(get_scaled_error(sigma_imp, sigma))
        end_time = time.time()
        runtimes.append(end_time - start_time)
        smae = get_smae(X_imp, X, X_masked)
        rmse = get_scaled_error(X_imp[:, :5], X[:, :5])
        smaes.append(smae)
        rmses.append(rmse)
    mean_rmse = np.mean(rmses, axis=0)
    mean_smae = np.mean(smaes, axis=0)
    # print("mean of scaled errors is: ")
    # print(np.mean(np.array(scaled_errors)))
    # print("std deviation of scaled errors is: ")
    # print(np.std(np.array(scaled_errors)))
    # print("\n")
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
    print("mean of rmses is: ")
    print(np.mean(np.array(rmses), axis=0))
    print("std deviation of rmses is: ")
    print(np.std(np.array(rmses), axis=0))
    print("\n")
    print("mean time for run is: ")
    print(np.mean(np.array(runtimes)))
