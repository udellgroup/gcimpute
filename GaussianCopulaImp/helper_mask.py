import numpy as np 

def mask_types(X, mask_num, seed):
    """
    Masks mask_num entries of the continuous, ordinal, and binary columns of X
    """
    if X.shape[1] % 3 != 0:
        raise NotImplementedError('Current implementation requires three types of variables have the same number of variables')
    X_masked = np.copy(X)
    mask_indices = []
    num_rows, num_cols = X_masked.shape
    num_cols_type = num_cols // 3
    np.random.seed(seed)
    for i in range(num_rows):
        for index_start in [0, num_cols_type, 2*num_cols_type]:
            rand_idx = np.random.choice(num_cols_type, mask_num, False) + index_start
            X_masked[i, rand_idx] = np.nan

    return X_masked

def mask(X, mask_fraction, seed=0, max_try=50):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    rng = np.random.default_rng(seed)
    count = 0
    X_masked = np.copy(X) 
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    p = X.shape[1]

    while True:
        mask_indices = obs_indices[rng.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
        # TO DO: use advanced indexing
        for i,j in mask_indices:
            X_masked[i,j] = np.nan
        # the masking is acceptable if there is no empty row
        if np.isnan(X_masked).sum(axis=1).max() < p:
            break
        else:
            count += 1
            X_masked = np.copy(X)
        if count == max_try:
            raise ValueError("Failure in masking data without empty rows")
    return X_masked

