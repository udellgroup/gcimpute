import numpy as np 
from scipy.stats import random_correlation, norm, expon

def _cont_to_ord(x, k, by = 'dist', seed=1):
    """
    convert entries of x to an ordinal with k levels using thresholds selected by one choice of the following:
    by = 'dist': select evenly spaced thresholds
    by = 'quantile': select random observations between .05 quantile and .95 quantile as thresholds
    by = 'sampling': select random samples from standard normal distribution as thresholds
    """
    # make the cutoffs based on the quantiles
    np.random.seed(seed)
    std_dev = np.std(x)
    if by == 'dist':
        cutoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    elif by == 'quantile':
        # samping cutoffs from 5% quantile to 95% quantile 
        select = (x>np.quantile(x, 0.05)) & (x<np.quantile(x, 0.95))
        cutoffs = np.random.choice(x[select], k-1, replace = False)
    elif by == 'sampling':
        # sampling from standard normal, does not depend on the input data
        cutoffs = np.random.normal(k-1)
    else:
        raise ValueError('Unsupported cutoff_by option')
    # TODO:
    # cuttoff = np.hstack((min_cutoff, cutoff, max_cutoff))
    # x = np.digitize(x, cuttoff)
    ords = np.zeros(len(x))
    for cutoff in cutoffs:
        ords += (x > cutoff).astype(int)
    
    return ords.astype(int) 


def cont_to_ord(x, k, by = 'dist', seed=1):
    """
    convert entries of x to an ordinal with k levels using thresholds selected by one choice of the following:
    by = 'dist': select evenly spaced thresholds
    by = 'quantile': select random observations between .05 quantile and .95 quantile as thresholds
    by = 'sampling': select random samples from standard normal distribution as thresholds
    """
    result = _cont_to_ord(x,k,by,seed)
    c = 0
    while len(np.unique(result))<k:
        c += 1
        result = _cont_to_ord(x,k,by,seed+c)
        if c==20:
            raise ValueError("Failure in generating cutoffs")
    return result

def generate_sigma(seed, p=15):
    np.random.seed(seed)
    W = np.random.normal(size=(p, p))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_LRGC(var_types, rank, sigma, n=500, ord_num=5, cont_transform=lambda x:x, seed=1, ordinalize_by='dist'):
    cont_index = var_types['cont']
    ord_index = var_types['ord']
    bin_index = var_types['bin']
    all_index = cont_index + ord_index + bin_index
    p = len(all_index)

    np.random.seed(seed)
    W = np.random.normal(size=(p,rank))
    # TODO: check everything of this form with APPLY
    for i in range(W.shape[0]):
        W[i,:] = W[i,:]/np.sqrt(np.sum(np.square(W[i,:]))) * np.sqrt(1 - sigma)
    Z = np.dot(np.random.normal(size=(n,rank)), W.T) + np.random.normal(size=(n,p), scale=np.sqrt(sigma))
    X_true = Z
    if len(cont_index)>0:
        X_true[:,cont_index] = cont_transform(X_true[:,cont_index])
    for ind in bin_index:
        X_true[:,ind] = cont_to_ord(Z[:,ind], k=2, by=ordinalize_by)
    for ind in ord_index:
         X_true[:,ind] = cont_to_ord(Z[:,ind], k=ord_num, by=ordinalize_by)
    return X_true, W


def generate_mixed_from_gc(sigma, n=2000, seed=1, var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}, cutoff_by='dist'):
    '''
    sigma: either a single correlation matrix or a list of correlation matrices
    '''
    if not isinstance(sigma, list):
        sigma = [sigma]
    cont_index = var_types['cont']
    ord_index = var_types['ord']
    bin_index = var_types['bin']
    all_index = cont_index + ord_index + bin_index
    p = len(all_index)
    if min(all_index)!=0 or max(all_index) != (p-1) or len(set(all_index)) != p:
        raise ValueError('Inconcistent specification of variable types indexing')
    if sigma[0].shape[1] != p :
        raise ValueError('Inconcistent dimension between variable lengths and copula correlation')
    np.random.seed(seed)
    X = np.concatenate([np.random.multivariate_normal(np.zeros(p), s, size=n) for s in sigma], axis=0)
    # marginal transformation
    X[:,cont_index] = expon.ppf(norm.cdf(X[:,cont_index]), scale = 3)
    for ind in ord_index:
        X[:,ind] = cont_to_ord(X[:,ind], k=5, by=cutoff_by)
    for ind in bin_index:
        X[:,ind] = cont_to_ord(X[:,ind], k=2, by=cutoff_by)
    return X

    
def get_mae(x_imp, x_true, x_obs=None):
    """
    gets Mean Absolute Error (MAE) between x_imp and x_true
    """
    if x_obs is not None:
        loc = np.isnan(x_obs) & (~np.isnan(x_true))
    else:
        loc = ~np.isnan(x_true)
    diff = x_imp[loc] - x_true[loc]
    return np.mean(np.abs(diff))


def get_rmse(x_imp, x_true, x_obs = None, relative=False):
    """
    gets Root Mean Squared Error (RMSE) or Normalized Root Mean Squared Error (NRMSE) between x_imp and x_true
    """
    if x_obs is not None:
        loc = np.isnan(x_obs) & (~np.isnan(x_true))
    else:
        loc = ~np.isnan(x_true)
    diff = x_imp[loc] - x_true[loc]
    #mse = np.mean(diff**2.0, axis=0)
    mse = np.mean(np.power(diff, 2))
    rmse = np.sqrt(mse)
    return rmse/np.sqrt(np.mean(np.power(x_true[loc],2))) if relative else rmse

        

def get_smae(x_imp, x_true, x_obs, 
             baseline=None, per_type=False, var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}):
    """
    gets Scaled Mean Absolute Error (SMAE) between x_imp and x_true
    """
    p = x_obs.shape[1]
    # the first column records the imputation error of x_imp,
    # while the second column records the imputation error of baseline
    error = np.zeros((p,2))

    # iterate over columns/variables
    for i, col in enumerate(x_obs.T):
        test = np.bitwise_and(~np.isnan(x_true[:,i]), np.isnan(col))
        # skip the column if there is no evaluation entry
        if np.sum(test) == 0:
            error[i,0] = np.nan
            error[i,1] = np.nan
            print(f'There is no entry to be evaluated in variable {col}.')
            continue
        
        base_imp = np.median(col[~np.isnan(col)]) if baseline is None else baseline[i]

        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        diff = np.abs(x_imp_col - x_true_col)
        base_diff = np.abs(base_imp - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1] = np.sum(base_diff)
        if error[i,1] == 0:
            print(f'Baseline imputation achieves zero imputation error in variable {i+1}.' 
                  f'There are {sum(test)} imputed entries, ranging from {x_true_col.min()} (min) to {x_true_col.max()} (max).')
            error[i,1] = np.nan

    if per_type:
        scaled_diffs = {}
        for name, val in var_types.items():
            scaled_diffs[name] = np.sum(error[name,0])/np.sum(error[name,1])
    else:
        scaled_diffs = error[:,0] / error[:,1]

    return scaled_diffs

def batch_iterable(X, batch_size=40):
    n = X.shape[0]
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield X[start:end]
        start = end

def get_smae_batch(x_imp, x_true, x_obs, 
                   batch_size = 40,
                   baseline=None, per_type=False, var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}):
    result = []
    baseline = np.nanmedian(x_obs,0) if baseline is None else baseline
    for imp, true, obs in zip(batch_iterable(x_imp,batch_size), batch_iterable(x_true,batch_size), batch_iterable(x_obs,batch_size)):
        scaled_diffs = get_smae(imp, true, obs, baseline=baseline, per_type=False)
        result.append(scaled_diffs)
    result = np.array(result)

    if per_type:
        scaled_diffs = {}
        for name, val in var_types.items():
            scaled_diffs[name] = np.nanmean(result[:,val], axis=1)
    else:
        scaled_diffs = result

    return scaled_diffs



def get_scaled_error(sigma_imp, sigma):
    """
    gets a scaled error between matrices |simga - sigma_imp|_F^2 / |sigma|_F^2
    """
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)


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

def mask(X, mask_fraction, seed=0, verbose=False):
    """
    Masks mask_fraction entries of X, raising a value error if an entire row is masked
    """
    complete = False
    count = 0
    X_masked = np.copy(X) 
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    while not complete:
        np.random.seed(seed)
        if verbose: print(seed)
        mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
        for i,j in mask_indices:
            X_masked[i,j] = np.nan
        complete = True
        for row in X_masked:
            if len(row[~np.isnan(row)]) == 0:
                seed += 1
                count += 1
                complete = False
                X_masked = np.copy(X)
                break
        if count == 50:
            raise ValueError("Failure in Masking data without empty rows")
    return X_masked, mask_indices, seed

def mask_per_row(X, seed=0, size=None, ratio=None):
    """
    Maskes one element uniformly at random from each row of X
    """
    if ratio is not None:
        size = int(X.shape[1] * ratio)
    X_masked = np.copy(X)
    n,p = X.shape
    for i in range(n):
        np.random.seed(seed*n+i)
        locs = np.arange(p)[~np.isnan(X[i,:])]
        if len(locs) <= size:
            raise ValueError("Size too large, empty row will appear!")
        rand_idx = np.random.choice(locs, size)
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

def grassman_dist(A,B):
    U1, d1, _ = np.linalg.svd(A, full_matrices = False)
    U2, d2, _ = np.linalg.svd(B, full_matrices = False)
    _, d,_ = np.linalg.svd(np.dot(U1.T, U2))
    theta = np.arccos(d)
    return np.linalg.norm(theta), np.linalg.norm(d1-d2)


