from GaussianCopulaImp.gaussian_copula import GaussianCopula
from GaussianCopulaImp.helper_data_generation import generate_sigma, generate_mixed_from_gc
from GaussianCopulaImp.helper_evaluation import get_smae, get_scaled_error
from GaussianCopulaImp.helper_mask import mask_types
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict
import sys
import argparse



def run_onerep(seed, n=2000, batch_size= 40, batch_c=0, max_iter=50, online=False, num_ord_updates=1,
			   var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
			   MASK_NUM=2, threshold=0.01, max_workers=4, cutoff_by='dist'):
	sigma = generate_sigma(seed, p=sum([len(value) for value in var_types.values()]))
	X = generate_mixed_from_gc(sigma=sigma, n=n, seed=seed, var_types=var_types, cutoff_by=cutoff_by)
	X_masked = mask_types(X, MASK_NUM, seed=seed)
	# model fitting
	
	p = sum([len(x) for x in var_types.values()])
	cont_indices = np.array([False] * p)
	cont_indices[var_types['cont']] = True
	start_time = time.time()
	if online:
		training_mode = 'minibatch-online'
		stepsize_func=lambda k, c=batch_c:c/(k+c)
	else:
		if batch_c>0:
			training_mode = 'minibatch-offline'
			stepsize_func=lambda k, c=batch_c:c/(k+c)
		else:
			training_mode = 'standard'
			stepsize_func = None
	gc = GaussianCopula(training_mode=training_mode, stepsize_func=stepsize_func, const_stepsize=None, batch_size=batch_size, 
		tol=threshold, max_iter=max_iter, 
		random_state=seed, n_jobs=max_workers)
	X_imp = gc.fit_transform(X_masked)
	sigma_imp = gc.get_params()['copula_corr']
	end_time = time.time()
	# save results 
	smae = get_smae(X_imp, X, X_masked)
	cor_error = get_scaled_error(sigma_imp, sigma)
	output = {'runtime':end_time - start_time, 'smae':smae, 'cor_error':cor_error}
	return output
	
def main(NUM_STEPS=10, n=2000, batch_size= 40, batch_c=0, max_iter=50, online=False, num_ord_updates=1,
		 var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
		 MASK_NUM=2, threshold=0.01, max_workers=4, cutoff_by='dist'):
	output_all = defaultdict(list)
	for i in tqdm(range(1, NUM_STEPS + 1)):
		output = run_onerep(
			seed=i, n=n, batch_size=batch_size, batch_c=batch_c, var_types=var_types, max_iter=max_iter, 
			online=online, num_ord_updates=num_ord_updates,
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
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--rep', default=10, type=int, help='number of repetitions to run')
	parser.add_argument('-w', '--workers', default=4, type=int, help='number of parallel workers to use')
	parser.add_argument('-s', '--bs', default=40, type=int, help='batch size')
	parser.add_argument('-c', '--bc', default=0, type=float, help='batch c')
	parser.add_argument('-i', '--iter', default=50, type=int, help='maximum number of iterations to run')
	parser.add_argument('--online', default=0, type=int, help='whether to perform online update')
	parser.add_argument('-o', '--ordupdate', default=1, type=int, help='number of oridinal updates in each EM iter')
	args = parser.parse_args()

	main(NUM_STEPS=args.rep, max_workers=args.workers, 
		batch_size=args.bs, batch_c=args.bc, max_iter=args.iter, online=args.online==1, num_ord_updates=args.ordupdate)


# Results for reference
#----------------------------------------------
# Run command $ python standard_copula_generated.py
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
# Run command $ python standard_copula_generated.py --bs 40 --bc 5 --iter 100
# Minibatch run 2 passes
#----------------------------------------------
# when cutoff_by is 'dist'
# Runtime in seconds: mean 8.80, std 1.15
# Relative correlation error: mean 0.165, std 0.015
# cont imputation SMAE: mean 0.796, std 0.016
# ord imputation SMAE: mean 0.795, std 0.025
# bin imputation SMAE: mean 0.660, std 0.033
#----------------------------------------------
# Run command $ python standard_copula_generated.py --bs 40 --bc 5 --iter 50
# Minibatch run 1 passes 
#----------------------------------------------
# when cutoff_by is 'dist'
# Runtime in seconds: mean 4.64, std 0.07
# Relative correlation error: mean 0.175, std 0.014
# cont imputation SMAE: mean 0.813, std 0.014
# ord imputation SMAE: mean 0.818, std 0.027
# bin imputation SMAE: mean 0.682, std 0.036
#-----------------------------------------------
# Run command $ python standard_copula_generated.py --bs 40 --bc 5 --iter 50 --online 1
# Online run 1 pass (batch_c = 5, batch_size = 40)
#----------------------------------------------
# when cutoff_by is 'dist'
# Runtime in seconds: mean 5.19, std 0.03
# Relative correlation error: mean 0.183, std 0.016
# cont imputation SMAE: mean 0.837, std 0.013
# ord imputation SMAE: mean 0.848, std 0.029
# bin imputation SMAE: mean 0.715, std 0.041




