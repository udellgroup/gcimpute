from GaussianCopulaImp.low_rank_gaussian_copula import LowRankGaussianCopula
from helpers import generate_LRGC, grassman_dist, mask, get_rmse
import numpy as np
import time
from tqdm import tqdm
import argparse
from collections import defaultdict

def run_onerep(setting, seed=1, 
               n=500, p=200, rank=10, noise_ratio=0.1, mask_ratio=0.4,
               threshold=0.01, max_iter=50, 
               ordinalize_by='quantile', ord_num=5):
    '''
    To be consistent with the experimental setting in the paper "Matrix Completion with Quantified Uncertainty through Low Rank Gaussian Copula", 
    we use ordinalized_by = 'quantile'. Setting ordinalized_by = 'dist' will yield an easier setting, since smaller MAE can be achieved.
    '''
    var_types = {'cont':[], 'ord':[], 'bin':[]}
    if setting == 'LR-cont':
        cont_transform = lambda x:x
        var_types['cont'] = list(range(p))
    elif setting == 'HR-cont':
        cont_transform = lambda x: np.power(x,3)
        var_types['cont'] = list(range(p))
    else:
        cont_transform = None
        var_t = 'ord' if  'ord' in setting else 'bin'
        var_types[var_t] = list(range(p))

    Xtrue, Wtrue = generate_LRGC(var_types=var_types, 
                                 rank=rank, sigma=noise_ratio, n=n, 
                                 cont_transform=cont_transform,
                                 seed=seed, 
                                 ordinalize_by=ordinalize_by, ord_num=ord_num)

    np.random.seed(seed)
    X_masked, mask_indices, _ = mask(Xtrue, mask_fraction = mask_ratio, seed=seed)

    start_time = time.time()
    LRGC = LowRankGaussianCopula()
    X_imp, W, sigma_est = LRGC.impute_missing(X=X_masked, rank=rank, threshold=threshold, max_iter=max_iter, seed=seed)
    end_time = time.time()

    if setting in ['ord', 'bin']:
        error = get_mae(X_imp, Xtrue, X_masked)
    else:
        error = get_rmse(X_imp, Xtrue, X_masked, relative=True)
    W_err = grassman_dist(W,Wtrue)
    return {'error':error, 'W_err':W_err, 'noise_ratio':sigma_est, 'runtime':end_time - start_time}

def main(setting, NUM_STEPS=10, 
         n=500, p=200,
         threshold=0.01, max_iter=50):
    if setting in ['LR-cont','HR-cont']:
        rank = 10
        mask_ratio = 0.4
        noise_ratio = 0.1
    elif setting in ['HighSNR-ord','HighSNR-bin']:
        rank = 5
        mask_ratio = 0.6
        noise_ratio = 0.1
    elif setting in ['LowSNR-ord','LowSNR-bin']:
        rank = 5
        mask_ratio = 0.6
        noise_ratio = 0.5
    else:
        raise ValueError('invalid setting value')

    output_all = defaultdict(list)
    for i in tqdm(range(1, NUM_STEPS + 1)):
        output = run_onerep(setting=setting, 
                            seed=i, threshold=threshold, max_iter=max_iter,
                            n=n, p=p, rank=rank, noise_ratio=noise_ratio, mask_ratio=mask_ratio)
        for name, value in output.items():
            output_all[name].append(value)
    # restults
    for name,value in output_all.items():
        output_all[name] = np.array(value)
    print(f"Runtime in seconds: mean {output_all['runtime'].mean():.2f}, std {output_all['runtime'].std():.2f}")
    print(f"Grassman distance of the subspace: mean {output_all['W_err'].mean():.3f}, std {output_all['W_err'].std():.3f}")
    print(f"Estimated subspace noise ratio (true value {noise_ratio}): mean {output_all['noise_ratio'].mean():.3f}, std {output_all['noise_ratio'].std():.3f}")
    error_type = 'NRMSE' if 'cont' in setting else 'MAE'
    print(f"Imputation error in {error_type}: mean {output_all['error'].mean():.3f}, std {output_all['error'].std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', default='LR-cont', type=str, 
        help='one of the following experimental settings to run: LR-cont, HR-cont, HighSNR-ord, LowSNR-ord, HighSNR-bin, LowSNR-bin')
    parser.add_argument('-r', '--rep', default=10, type=int, help='number of repetitions to run')
    parser.add_argument('-i', '--iter', default=50, type=int, help='maximum number of iterations to run')
    parser.add_argument('-t', '--threshold', default=0.01, type=float, help='minimal parameter difference for model update')

    args = parser.parse_args()

    main(setting=args.setting, NUM_STEPS=args.rep, max_iter=args.iter, threshold=args.threshold)

#  Results for reference

# -------------------------------------
# LOW RANK CONTINUOUS DATA
# -------------------------------------
# python lowrank_copula_generated.py 
# Standard run for LR-cont
# -------------------------------------
# Runtime in seconds: mean 1.22, std 0.04
# Grassman distance of the subspace: mean 0.292, std 0.090
# Estimated subspace noise ratio (true value 0.1): mean 0.102, std 0.001
# Imputation error in NRMSE: mean 0.347, std 0.002
# -------------------------------------
# python lowrank_copula_generated.py -t 0.001
# using higher precision for LR-cont 
# -------------------------------------
# Runtime in seconds: mean 2.52, std 0.22
# Grassman distance of the subspace: mean 0.293, std 0.090
# Estimated subspace noise ratio (true value 0.1): mean 0.101, std 0.001
# Imputation error in NRMSE: mean 0.347, std 0.002

# -------------------------------------
# HIRH RANK CONTINUOUS DATA
# -------------------------------------
# python lowrank_copula_generated.py 
# Standard run for HR-cont
# ----------------------------------------
# Runtime in seconds: mean 1.31, std 0.06
# Grassman distance of the subspace: mean 0.292, std 0.090
# Estimated subspace noise ratio (true value 0.1): mean 0.102, std 0.001
# Imputation error in NRMSE: mean 0.517, std 0.005
# -------------------------------------
# python lowrank_copula_generated.py -s HR-cont -t 0.001
# using higher precision for HR-cont 
# -------------------------------------
# Runtime in seconds: mean 2.22, std 0.02
# Grassman distance of the subspace: mean 0.293, std 0.090
# Estimated subspace noise ratio (true value 0.1): mean 0.101, std 0.001
# Imputation error in NRMSE: mean 0.517, std 0.005


# -------------------------------------
# 1-5 ORDINAL DATA
# -------------------------------------
# The obtained imputation error is much smaller than the recorded numbers in the paper "Matrix Completion with Quantified Uncertainty through Low Rank Gaussian Copula".
# The reason may still be how the cutoffs for ordinal variables are generated. 
#
# Runtime in seconds: mean 99.65, std 3.66
# Grassman distance of the subspace: mean 0.256, std 0.099
# Estimated subspace noise ratio (true value 0.1): mean 0.126, std 0.002
# Imputation error in MAE: mean 0.291, std 0.005

