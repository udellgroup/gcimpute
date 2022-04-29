import numpy as np 

def mask_types(X, mask_num, seed, var_types):
    """
    Masks mask_num entries of the continuous, ordinal, and binary columns of X
    """
    X = np.asarray(X)
    p = sum([len(x) for x in var_types.values()])
    if X.shape[1] != p:
        print('Inconsistent data and var types')
        raise
    X_masked = np.copy(X)
    mask_indices = []
    num_rows, num_cols = X_masked.shape
    num_cols_type = num_cols // 3
    rng = np.random.default_rng(seed)
    for _type, _index in var_types.items():
        for i, row in enumerate(X_masked):
            if len(_index)>0:
                rand_idx = rng.choice(_index, mask_num, False)
                row[rand_idx] = np.nan

    return X_masked
    

def mask_MCAR(X, mask_fraction, seed=1, allow_empty_row=False, silence_rate=0.01):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    X_masked = np.copy(X) 
    obs_coors = np.argwhere(~np.isnan(X))
    n_obs = obs_coors.shape[0]
    n,p = X.shape

    mask_indices = rng.choice(n_obs, size=int(mask_fraction*n_obs), replace=False)
    mask_coors = obs_coors[mask_indices]
    X_masked[mask_coors[:,0], mask_coors[:,1]] = np.nan
    which_empty = np.flatnonzero((~np.isnan(X_masked)).sum(axis=1) == 0) 
    if not allow_empty_row:
        for row in which_empty:
            obs_loc = np.flatnonzero(~np.isnan(X[row,:]))
            index = rng.choice(obs_loc, size=1)
            X_masked[row,index] = X[row,index]

        r = (np.isnan(X_masked).sum() - (n*p-n_obs))/n_obs
        if r<mask_fraction-silence_rate:
            print(f'Actual masking ratio: {np.round(r, 4)}')
    return X_masked
    """
    count = 0
    while True:
        # the masking is acceptable if there is no empty row
        if allow_empty_row or np.isnan(X_masked).sum(axis=1).max() < p:
            break
        else:
            count += 1
            X_masked = np.copy(X)
        if count == max_try:
            raise ValueError("Failure in masking data without empty rows")
    return X_masked
    """

def split_mask_val_test(X_mask, X, val_ratio = 0.5, rng = None, seed = 1):
    if rng is None:
        rng = np.random.default_rng(seed)
    mask_coors = np.argwhere(~np.isnan(X) & np.isnan(X_mask))
    n = len(mask_coors)
    index_val = rng.choice(np.arange(n), size=int(n*val_ratio), replace=False)
    index_test = np.setdiff1d(np.arange(n), index_val)
    r = {'validation':X_mask.copy(), 'test':X_mask.copy()}
    loc = {'validation':index_val, 'test':index_test}
    for name,v in loc.items():
        r[name][mask_coors[v,0], mask_coors[v,1]] = X[mask_coors[v,0], mask_coors[v,1]]
    return r
    

