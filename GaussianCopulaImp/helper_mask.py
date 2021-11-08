import numpy as np 

def mask_types(X, mask_num, seed, var_types):
    """
    Masks mask_num entries of the continuous, ordinal, and binary columns of X
    """
    p = sum([len(x) for x in var_types.values()])
    if X.shape[1] != p:
        print('Inconsistent data and var types')
        raise
    X_masked = np.copy(X)
    mask_indices = []
    num_rows, num_cols = X_masked.shape
    num_cols_type = num_cols // 3
    np.random.seed(seed)
    for _type, _index in var_types.items():
        for i, row in enumerate(X_masked):
            if len(_index)>0:
                rand_idx = np.random.choice(_index, mask_num, False)
                row[rand_idx] = np.nan

    return X_masked

def mask_MCAR(X, mask_fraction, seed=1, max_try=50):
    '''
    Randomly mask a proportion of observed entries by sampling without replacement.
    Args:
        X: array like of shape (n_samples, n_features)
            The data to be masked. Can be incomplete itself. 
        ratio: float in (0,1)
            The masking is done by sampling from all observed entries without replacement.
    '''
    return mask(X, mask_fraction, seed, max_try)
    
    

def mask(X, mask_fraction, seed=0, max_try=50):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    if hasattr(X, 'columns'):
        columns = X.columns
        X = np.array()
    rng = np.random.default_rng(seed)
    count = 0
    X_masked = np.copy(X) 
    obs_coors = np.argwhere(~np.isnan(X))
    n_obs = obs_coors.shape[0]
    p = X.shape[1]

    while True:
        mask_indices = rng.choice(n_obs, size=int(mask_fraction*n_obs), replace=False)
        mask_coors = obs_coors[mask_indices]
        X_masked[mask_coors[:,0], mask_coors[:,1]] = np.nan
        # the masking is acceptable if there is no empty row
        if np.isnan(X_masked).sum(axis=1).max() < p:
            break
        else:
            count += 1
            X_masked = np.copy(X)
        if count == max_try:
            raise ValueError("Failure in masking data without empty rows")
    return X_masked

