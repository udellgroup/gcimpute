import numpy as np 

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
        cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    elif by == 'quantile':
        # samping cutoffs from 5% quantile to 95% quantile 
        select = x>np.quantile(x, 0.05) & x<np.quantile(0.95)
        cutoffs = np.random.choice(x[select], k-1, replace = False)
    elif by == 'sampling':
        # sampling from standard normal, does not depend on the input data
        cutoffs = np.random.normal(k-1)
    # TODO:
    # cuttoff = np.hstack((min_cutoff, cutoff, max_cutoff))
    # x = np.digitize(x, cuttoff)
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    
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

def generate_sigma(seed):
    np.random.seed(seed)
    W = np.random.normal(size=(15, 15))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_LRGC(rank, sigma, n=500, p_seq=(100,100,100), ord_num=5, cont_type = 'LR', seed=1):
    cont_indices = None
    bin_indices = None
    ord_indices = None
    if p_seq[0] > 0:
        cont_indices = range(p_seq[0])
    if p_seq[1] > 0:
        ord_indices = range(p_seq[0],p_seq[0] + p_seq[1])
    if p_seq[2] > 0:
        bin_indices = range(p_seq[0] + p_seq[1], p_seq[0] + p_seq[1] + p_seq[2])
    p = np.sum(p_seq)
    np.random.seed(seed)
    W = np.random.normal(size=(p,rank))
    # TODO: check everything of this form with APPLY
    for i in range(W.shape[0]):
        W[i,:] = W[i,:]/np.sqrt(np.sum(np.square(W[i,:]))) * np.sqrt(1 - sigma)
    Z = np.dot(np.random.normal(size=(n,rank)), W.T) + np.random.normal(size=(n,p), scale=np.sqrt(sigma))
    X_true = Z
    if cont_indices is not None:
        if cont_type != 'LR':
            X_true[:,cont_indices] = X_true[:,cont_indices]**3
    if bin_indices is not None:
        for bin_index in bin_indices:
            X_true[:,bin_index] = continuous2ordinal(Z[:,bin_index], k=2)
    if ord_indices is not None:
        for ord_index in ord_indices:
            X_true[:,ord_index] = continuous2ordinal(Z[:,ord_index], k=ord_num)
    return X_true, W
    
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
    return rmse if not relative else rmse/np.sqrt(np.mean(np.power(x_true[loc],2)))
        

def get_smae(x_imp, x_true, x_obs, Med=None, per_type=False, cont_loc=None, bin_loc=None, ord_loc=None):
    """
    gets Scaled Mean Absolute Error (SMAE) between x_imp and x_true
    """
    error = np.zeros((x_obs.shape[1],2))
    for i, col in enumerate(x_obs.T):
        test = np.bitwise_and(~np.isnan(x_true[:,i]), np.isnan(col))
        if np.sum(test) == 0:
            error[i,0] = np.nan
            error[i,1] = np.nan
            continue
        col_nonan = col[~np.isnan(col)]
        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        if Med is not None:
            median = Med[i]
        else:
            median = np.median(col_nonan)
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1]= np.sum(med_diff)
    if per_type:
        if not cont_loc:
            cont_loc = [True] * 5 + [False] * 10
        if not bin_loc:
            bin_loc = [False] * 5 + [True] * 5 + [False] * 5 
        if not ord_loc:
            ord_loc = [False] * 10 + [True] * 5
        loc = [cont_loc, bin_loc, ord_loc]
        scaled_diffs = np.zeros(3)
        for j in range(3):
            scaled_diffs[j] = np.sum(error[loc[j],0])/np.sum(error[loc[j],1])
    else:
        scaled_diffs = error[:,0] / error[:,1]
    return scaled_diffs

def get_smae_per_type(x_imp, x_true, x_obs, cont_loc=None, bin_loc=None, ord_loc=None):
    if not cont_loc:
        cont_loc = [True] * 5 + [False] * 10
    if not bin_loc:
        bin_loc = [False] * 5 + [True] * 5 + [False] * 5 
    if not ord_loc:
        ord_loc = [False] * 10 + [True] * 5
    loc = [cont_loc, bin_loc, ord_loc]
    scaled_diffs = np.zeros(3)
    for j in range(3):
        missing = np.isnan(x_obs[:,loc[j]])
        med = np.median(x_obs[:,loc[j]][~missing])
        diff = np.abs(x_imp[:,loc[j]][missing] - x_true[:,loc[j]][missing])
        med_diff = np.abs(med - x_true[:,loc[j]][missing])
        scaled_diffs[j] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs
    
def get_smae_per_type_online(x_imp, x_true, x_obs, Med):
    for i, col in enumerate(x_obs.T):
        missing = np.isnan(col)
        x_true_col = x_true[np.isnan(col),i]
        x_imp_col = x_imp[np.isnan(col),i]
        median = Med[i]
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        scaled_diffs[i] = np.sum(diff)/np.sum(med_diff)
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
            mask_indices.append((i,idx+num_cols // 3))
        for idx in rand_idx[2*mask_num:]:
            X_masked[i, idx+10] = np.nan
            mask_indices.append((i,idx+num_cols // 3 * 2))
    return X_masked, mask_indices

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
        if (verbose): print(seed)
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


