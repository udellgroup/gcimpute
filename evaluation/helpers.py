import numpy as np

def cont_to_binary(x):
    """
    convert entries of x to binary using a random threshold function
    """
    # make the cuttoff a random sample and ensure at least 10% are in each class
    while True:
        cutoff = np.random.choice(x)    
        if len(x[x < cutoff]) > 0.1*len(x) and len(x[x < cutoff]) < 0.9*len(x):
            break
    return (x > cutoff).astype(int)

def cont_to_ord(x, k):
    """
    convert entries of x to an ordinal with k levels using evenenly space thresholds
    """
    # make the cutoffs based on the quantiles
    if k == 2:
        return cont_to_binary(x)
    std_dev = np.std(x)
    cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    return ords.astype(int)

def get_mae(x_imp, x_true):
    """
    gets Mean Absolute Error (MAE) between x_imp and x_true
    """
    return np.mean(np.abs(x_imp - x_true))
        

def get_smae(x_imp, x_true, x_obs):
    """
    gets Scaled Mean Absolute Error (SMAE) between x_imp and x_true
    """
    scaled_diffs = np.zeros(x_obs.shape[1])
    for i, col in enumerate(x_obs.T):
        col_nonan = col[~np.isnan(col)]
        x_true_col = x_true[np.isnan(col),i]
        x_imp_col = x_imp[np.isnan(col),i]
        median = np.median(col_nonan)
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        scaled_diffs[i] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs

def get_rmse(x_imp, x_true):
    """
    gets Root Mean Squared Error (RMSE) between x_imp and x_true
    """
    diff = x_imp - x_true
    mse = np.mean(diff**2.0, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def get_scaled_error(sigma_imp, sigma):
    """
    gets a scaled error between matrices |simga - sigma_imp|_F^2 / |sigma|_F^2
    """
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)


def mask_types(X, mask_num, seed):
    """
    Masks mask_num entries of the continuous, ordinal, and binary columns of X
    """
    X_masked = np.copy(X)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i) # uncertain if this is necessary
        rand_idx = np.concatenate((np.random.choice(num_cols // 3, mask_num, False), np.random.choice(num_cols // 3, mask_num, False), np.random.choice(num_cols // 3, mask_num, False)))
        for idx in rand_idx[:mask_num]:
            X_masked[i, idx] = np.nan
            mask_indices.append((i,idx))
        for idx in rand_idx[mask_num:2*mask_num]:
            X_masked[i, idx+5] = np.nan
            mask_indices.append((i,idx+5))
        for idx in rand_idx[2*mask_num:]:
            X_masked[i, idx+10] = np.nan
            mask_indices.append((i,idx+10))
    return X_masked, mask_indices

def mask(X, mask_fraction, seed=0):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    X_masked = np.copy(X) 
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    np.random.seed(seed)
    mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
    for i,j in mask_indices:
        X_masked[i,j] = np.nan
        row = X_masked[i,:]
        if len(row[~np.isnan(row)]) == 0:
            print(i)
            raise ValueError("Failure in Generation, row is entirely nan")
    return X_masked, mask_indices

def mask_one_per_row(X, seed=0):
    """
    Maskes one element uniformly at random from each row of X
    """
    X_masked = np.copy(X)
    n,p = X.shape
    for i in range(n):
        np.random.seed(seed*n+i)
        rand_idx = np.random.choice(p)
        X_masked[i,rand_idx] = np.nan
    return X_masked

def _project_to_correlation(covariance):
        """
        Projects a covariance to a correlation matrix, normalizing it's diagonal entries

        Args:
            covariance (matrix): a covariance matrix

        Returns:
            correlation (matrix): the covariance matrix projected to a correlation matrix
        """
        D = np.diagonal(covariance)
        D_neg_half = np.diag(1.0/np.sqrt(D))
        return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)


