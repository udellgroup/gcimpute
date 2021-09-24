from GaussianCopulaImp.expectation_maximization import ExpectationMaximization
from helpers import generate_sigma, generate_mixed_from_gc, mask_types, get_smae, get_scaled_error, get_smae_batch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm
from collections import defaultdict
import sys
import argparse

def run_onerep(seed=1, n=2000, 
	           batch_size= 40, batch_c=0, max_iter=50, const_decay=0.5,
	           num_ord_updates=1, threshold=0.01, max_workers=4,
			   var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
			   MASK_NUM=2,  cutoff_by='dist'):
	sigma = [generate_sigma(seed+i, p=sum([len(value) for value in var_types.values()])) for i in range(3)]
	X = generate_mixed_from_gc(sigma=sigma, n=n, seed=seed, var_types=var_types, cutoff_by=cutoff_by)
	X_masked = mask_types(X, MASK_NUM, seed=seed)

	# online model fitting 
	p = sum([len(val) for val in var_types.values()])
	cont_indices = np.zeros(p)
	cont_indices[var_types['cont']] = 1
	var_types_input = {'cont':cont_indices==1, 'ord':cont_indices==0}
	em = ExpectationMaximization(var_types = var_types_input)
	out = em.impute_missing_online(X=X_masked, 
		                           batch_size=batch_size, const_decay = const_decay,
		                           num_ord_updates=num_ord_updates,threshold=threshold, max_workers=max_workers)
	X_imp_online, copula_corr_change = out['imputed_data'], out['copula_corr_change']

	# offline model fitting
	em = ExpectationMaximization()
	out = em.impute_missing(X=X_masked, 
		    			    max_iter=max_iter,batch_size=batch_size, batch_c=batch_c, 
		                    num_ord_updates=num_ord_updates, threshold=threshold, max_workers=max_workers)
	X_imp_offline = out['imputed_data']
	
	# save results 
	smae_online = get_smae_batch(X_imp_online, X, X_masked, batch_size=batch_size, per_type=True, var_types=var_types)
	smae_offline = get_smae_batch(X_imp_offline, X, X_masked, batch_size=batch_size, per_type=True, var_types=var_types)
	output = {'copula_corr_change':pd.DataFrame(copula_corr_change), 'smae_online':pd.DataFrame(smae_online), 'smae_offline':pd.DataFrame(smae_offline)}
	return output


def main(NUM_STEPS=10, 
		 batch_size=40, batch_c=0, max_iter=50, const_decay=0.5,max_workers=4, 
		 var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))},
		 threshold=0.01, n=2000, MASK_NUM=2, cutoff_by='dist', num_ord_updates=1):
	output_all = defaultdict(list)
	for i in tqdm(range(1, NUM_STEPS + 1)):
		output = run_onerep(seed=i, n=n, 
							batch_size=batch_size, batch_c=batch_c, max_iter=max_iter, const_decay=const_decay,
							num_ord_updates=num_ord_updates, threshold=threshold, max_workers=max_workers, 
							var_types=var_types, MASK_NUM=MASK_NUM, cutoff_by=cutoff_by)
		for name, value in output.items():
			output_all[name].append(value)
	# restults
	output_all['copula_corr_change'] = sum(output_all['copula_corr_change'])/NUM_STEPS
	output_all['smae_online'] = sum(output_all['smae_online'])/NUM_STEPS
	output_all['smae_offline'] = sum(output_all['smae_offline'])/NUM_STEPS
	
	mpl.use('tkagg')
	fig, ax = plt.subplots(1,3, figsize=(12,3))
	output_all['copula_corr_change'].plot(ax=ax[0], title='copula change test statistics')
	output_all['smae_online'].plot(ax=ax[1], title='online imputation error')
	output_all['smae_offline'].plot(ax=ax[2], title='offline imputation error')
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--rep', default=10, type=int, help='number of repetitions to run')
	parser.add_argument('-s', '--bs', default=40, type=int, help='batch size')
	parser.add_argument('-c', '--bc', default=5, type=float, help='batch c (for offline minibatch)')
	parser.add_argument('-d', '--decay', default=0.5, type=float, help='constant decay rate (for online)')
	parser.add_argument('-w', '--workers', default=4, type=int, help='number of parallel workers to use')
	parser.add_argument('-i', '--iter', default=300, type=int, help='maximum number of iterations to run (for offline minibatch)')
	parser.add_argument('-o', '--ordupdate', default=1, type=int, help='number of oridinal updates in each EM iter')
	args = parser.parse_args()

	main(NUM_STEPS=args.rep, batch_size=args.bs, batch_c=args.bc, max_iter=args.iter, const_decay=args.decay,
		max_workers=args.workers, num_ord_updates=args.ordupdate)