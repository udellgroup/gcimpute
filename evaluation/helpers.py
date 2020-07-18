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

def mask(X, mask_fraction):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    X_masked = np.copy(X) 
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
    for i,j in mask_indices:
        X_masked[i,j] = np.nan
        row = X_masked[i,:]
        if len(row[~np.isnan(row)]) == 0:
            print(i)
            raise ValueError("Failure in Generation, row is entirely nan")
    return X_masked, mask_indices

def mask_one_per_row(X):
    """
    Maskes one element uniformly at random from each row of X
    """
    X_masked = np.copy(X)
    for i in range(X_masked.shape[0]):
        rand_idx = np.random.choice(X.shape[1])
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


