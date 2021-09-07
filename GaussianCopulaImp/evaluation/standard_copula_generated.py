from GaussianCopulaImp.em.expectation_maximization import ExpectationMaximization
from GaussianCopulaImp.evaluation.helpers import generate_sigma, generate_mixed_from_gc, mask_types, get_smae, get_scaled_error
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict
import sys

def run_onerep(seed, n=2000, batch_size= 40, batch_c=0, max_iter=50,
               var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
               MASK_NUM=2, threshold=0.01, max_workers=4, cutoff_by='dist'):
    sigma = generate_sigma(seed, p=sum([len(value) for value in var_types.values()]))
    X = generate_mixed_from_gc(sigma=sigma, n=n, seed=seed, var_types=var_types, cutoff_by=cutoff_by)
    X_masked = mask_types(X, MASK_NUM, seed=seed)
    # model fitting
    em = ExpectationMaximization()
    start_time = time.time()
    X_imp, sigma_imp = em.impute_missing(X=X_masked, threshold=threshold, max_iter=max_iter, max_workers=max_workers, batch_size=batch_size, batch_c=batch_c)
    end_time = time.time()
    # save results 
    smae = get_smae(X_imp, X, X_masked)
    cor_error = get_scaled_error(sigma_imp, sigma)
    output = {'runtime':end_time - start_time, 'smae':smae, 'cor_error':cor_error}
    return output
    
def main(NUM_STEPS=10, n=2000, batch_size= 40, batch_c=0, max_iter=50,
         var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
         MASK_NUM=2, threshold=0.01, max_workers=4, cutoff_by='dist'):
    output_all = defaultdict(list)
    for i in tqdm(range(1, NUM_STEPS + 1)):
        output = run_onerep(seed=i, n=n, batch_size=batch_size, batch_c=batch_c, var_types=var_types, max_iter=max_iter,
                            MASK_NUM=MASK_NUM, threshold=threshold, max_workers=max_workers, cutoff_by=cutoff_by)
        for name, value in output.items():
            output_all[name].append(value)
    # restults
    for name,value in output_all.items():
        output_all[name] = np.array(value)
    print(f"Runtime in seconds: mean {output_all['runtime'].mean():.2f}, std {output_all['runtime'].std():.2f}")
    print(f"Relative correlation error: mean {output_all['cor_error'].mean():.3f}, std {output_all['cor_error'].std():.3f}")
    mean_smaes = np.mean(output_all['smae'],axis=0)
    std_smaes = np.std(output_all['smae'],axis=0)
    for name,value in var_types.items():
        print(f'{name} imputation SMAE: mean {mean_smaes[value].mean():.3f}, std {mean_smaes[value].std():.3f}')

if __name__ == "__main__":
    NUM_STEPS = 10 if len(sys.argv) == 1 else int(sys.argv[1])
    max_workers = 4 if len(sys.argv) <=2 else int(sys.argv[2])
    batch_size = 40 if len(sys.argv) <=3 else int(sys.argv[3])
    batch_c = 0 if len(sys.argv) <=4 else int(sys.argv[4])
    max_iter = 50 if len(sys.argv) <=5 else int(sys.argv[5])
    main(NUM_STEPS=NUM_STEPS, max_workers=max_workers, batch_size=batch_size, batch_c=batch_c, max_iter=max_iter)


# Results for reference
#----------------------------------------------
# Standard run:
#----------------------------------------------
# when cutoff_by is 'dist'
# Runtime in seconds: mean 22.47, std 0.91
# Relative correlation error: mean 0.165, std 0.015
# cont imputation SMAE: mean 0.796, std 0.017
# ord imputation SMAE: mean 0.797, std 0.028
# bin imputation SMAE: mean 0.664, std 0.035
# when cutoff_by is 'quantile'
# Runtime in seconds: mean 24.51, std 1.79
# Relative correlation error: mean 0.199, std 0.030
# cont imputation SMAE: mean 0.804, std 0.019
# ord imputation SMAE: mean 0.731, std 0.015
# bin imputation SMAE: mean 0.835, std 0.062
#----------------------------------------------
# Minibatch run (batch_c = 5, batch_size = 40, max_iter = 100)
#----------------------------------------------
# when cutoff_by is 'dist'
# Runtime in seconds: mean 8.80, std 1.15
# Relative correlation error: mean 0.165, std 0.015
# cont imputation SMAE: mean 0.796, std 0.016
# ord imputation SMAE: mean 0.795, std 0.025
# bin imputation SMAE: mean 0.660, std 0.033





