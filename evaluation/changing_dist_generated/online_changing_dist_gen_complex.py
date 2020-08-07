from em.online_expectation_maximization import OnlineExpectationMaximization
import numpy as np
from evaluation.helpers import cont_to_binary, cont_to_ord, get_smae, get_rmse, get_scaled_error, mask, mask_one_per_row
from transforms.online_ordinal_marginal_estimator import OnlineOrdinalMarginalEstimator
from statsmodels.distributions.empirical_distribution import ECDF


def get_stats():
    NUM_RUNS = 10
    NUM_SAMPLES = 10000
    BATCH_SIZE = 50
    list_of_smaes = []
    list_of_rmses = []
    list_of_scaled_errors = []
    for i in range(NUM_RUNS):
        print("starting epoch: " + str(i + 1))
        scaled_errors = []
        np.random.seed(i)
        sigma1 = np.array(
            [
                [1, 0.339135, 0.326585],
                [0.339135, 1, -0.778398],
                [0.326585, -0.778398, 1]
            ]
        )
        sigma2 = np.array(
            [
                [1, -0.778398, 0.339135],
                [-0.778398, 1, 0.326585],
                [0.339135, 0.326585, 1]
            ]
        )
        sigma3 = np.array(
            [
                [1, 0.326585, -0.778398],
                [0.326585, 1, 0.339135],
                [-0.778398, 0.339135, 1]
            ]
        )
        mean = np.array([0.0, 0.0, 0.0])
        X1 = np.random.multivariate_normal(mean, sigma1, size=NUM_SAMPLES)
        X2 = np.random.multivariate_normal(mean, sigma2, size=NUM_SAMPLES)
        X3 = np.random.multivariate_normal(mean, sigma3, size=NUM_SAMPLES)
        X = np.vstack((X1, X2, X3))
        X[:, 1] = cont_to_ord(X[:, 1], k=5)
        X[:, 2] = cont_to_binary(X[:, 2])
        X_masked = mask_one_per_row(X)
        oem = OnlineExpectationMaximization(cont_indices=np.array(
            [True, False, False]), ord_indices=np.array([False, True, True]))
        X_imp = np.empty(X.shape)
        i = 0
        while i*BATCH_SIZE < X_masked.shape[0]:
            X_imp_row, sigma_imp = oem.partial_fit_and_predict(
                X_masked[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :])
            X_imp[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = X_imp_row
            smae = get_smae(X_imp[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], X[i*BATCH_SIZE:(
                i+1)*BATCH_SIZE, :], X_masked[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :])
            i += 1
        smae = get_smae(X_imp, X, X_masked)
        rmse = get_scaled_error(X_imp[:, 0], X[:, 0])
        list_of_smaes.append(smae)
        list_of_rmses.append(rmse)
    mean_rmse = np.mean(list_of_rmses, axis=0)
    mean_smae = np.mean(list_of_smaes, axis=0)
    print("mean smae")
    print(mean_smae)
    print("mean rmse")
    print(mean_rmse)


if __name__ == "__main__":
    get_stats()
