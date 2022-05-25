import numpy as np

def observed_part(x): 
	return x[~np.isnan(x)]

def unique_observed(x):
	return np.unique(observed_part(x))

def num_unique(x):
	return len(unique_observed(x))

def dict_values_len(d): 
	return sum([len(v) for v in d.values()])

def merge_dict_arrays(d):
	return np.concatenate(list(d.values()), axis=0)


