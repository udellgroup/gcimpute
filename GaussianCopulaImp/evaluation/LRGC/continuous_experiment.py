from em.low_rank_expectation_maximization import LowRankExpectationMaximization
from evaluation.helpers import generate_LRGC, grassman_dist, mask, get_rmse, get_relative_rmse
import numpy as np

def continuous_experiment():
    n = 500
    p = 200
    rank = 10
    sigma = 0.1
    ratio = 0.4


    TRIALS = 10
    nrmses = []
    Ws = []
    sigmas = []
    for i in range(TRIALS):
        print("trial is: " + str(i + 1))
        # LOW RANK
        Xtrue, Wtrue = generate_LRGC(rank=rank, sigma=sigma, n=n, p_seq=(p,0,0), cont_type = 'LR', seed = i)
        np.random.seed(i)
        X_masked, mask_indices, _ = mask(Xtrue, ratio)
        LREM = LowRankExpectationMaximization()
        X_imp, W, sigma_est = LREM.impute_missing(X_masked, rank)
        nrmse = get_relative_rmse(X_imp, Xtrue, X_masked)
        print("nrmse is: " + str(nrmse))
        nrmses.append(nrmse)
        err = grassman_dist(W,Wtrue)
        print('subspace distance is '+str(err))
        Ws.append(err)
        print('sigma is '+str(sigma_est))
        sigmas.append(sigma_est)
    #print("All rmses are: ")
    #print(rmses)
    print("mean of nrmses is: ")
    #mean_rmses = np.mean(np.array(rmses),axis=0)
    #print(mean_rmses)
    #print("mean of mean of rmses is: ")
    print(np.mean(nrmses))
    
    return nrmses, sigmas, Ws


if __name__ == "__main__":
    r, s, w = continuous_experiment()




