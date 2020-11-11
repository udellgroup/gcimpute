from em.low_rank_expectation_maximization import LowRankExpectationMaximization
from evaluation.helpers import generate_LRGC, grassman_dist, mask, get_mae
import numpy as np



def ordinal_experiment():
    n = 500
    p = 200
    rank = 5
    sigma = 0.1
    ratio = 0.6


    TRIALS = 5
    error = []
    Ws = []
    sigmas = []
    for i in range(TRIALS):
        print("trial is: " + str(i + 1))
        # LOW RANK
        Xtrue, Wtrue = generate_LRGC(rank = rank, sigma = sigma, n=n, p_seq=(0,p,0), seed=i)
        np.random.seed(i)
        X_masked, mask_indices, _ = mask(Xtrue, ratio)
        LREM = LowRankExpectationMaximization()
        X_imp, W, sigma_est = LREM.impute_missing(X_masked, rank, verbose=True)
        error.append(get_mae(X_imp, Xtrue, X_masked))
        print("MAE is: " + str(error[-1]))
        err = grassman_dist(W,Wtrue)
        print('subspace distance is '+str(err))
        Ws.append(err)
        print('sigma is '+str(sigma_est))
        sigmas.append(sigma_est)
    #print("All rmses are: ")
    #print(rmses)
    #print(mean_rmses)
    #print("mean of mean of rmses is: ")
    #print(np.mean(rmses))
    
    return error, sigmas, Ws


if __name__ == "__main__":
    r, s,w = ordinal_experiment()
    print("Mean MAE is: ", np.mean(r))



