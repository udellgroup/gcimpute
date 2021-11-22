import numpy as np 
from scipy.stats import random_correlation, norm, expon
from scipy.linalg import svdvals

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
    if not relative:
        return rmse
    else:
        # RMSE of zero-imputation
        norm = np.sqrt(np.mean(np.power(x_true[loc],2)))
        return rmse/norm

def get_smae(x_imp, x_true, x_obs, 
             baseline=None, per_type=False, 
             var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}):
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
            print(f'There is no entry to be evaluated in variable {i}.')
            continue
        
        base_imp = np.median(col[~np.isnan(col)]) if baseline is None else baseline[i]

        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        diff = np.abs(x_imp_col - x_true_col)
        base_diff = np.abs(base_imp - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1] = np.sum(base_diff)
        if error[i,1] == 0:
            print(f'Baseline imputation achieves zero imputation error in variable {i+1}.') 
            print(f'The {sum(test)} true values range from {x_true_col.min()} (min) to {x_true_col.max()} (max).')
            error[i,1] = np.nan

    if per_type:
        scaled_diffs = {}
        for name, val in var_types.items():
            scaled_diffs[name] = np.sum(error[name,0])/np.sum(error[name,1])
    else:
        scaled_diffs = error[:,0] / error[:,1]

    return scaled_diffs

def batch_iterable(X, batch_size=40):
    '''
    Generator which returns a mini-batch view of X.
    '''
    n = X.shape[0]
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield X[start:end]
        start = end

def get_smae_batch(x_imp, x_true, x_obs, 
                   batch_size = 40,
                   baseline=None, per_type=False, 
                   var_types = {'cont':list(range(5)), 'ord':list(range(5, 10)), 'bin':list(range(10, 15))}):
    '''
    Compute SMAE in the unit of a mini-batch
    '''
    result = []
    baseline = np.nanmedian(x_obs,0) if baseline is None else baseline
    for imp, true, obs in zip(batch_iterable(x_imp,batch_size), batch_iterable(x_true,batch_size), batch_iterable(x_obs,batch_size)):
        scaled_diffs = get_smae(imp, true, obs, baseline=baseline, per_type=False)
        result.append(scaled_diffs)
    result = np.array(result)

    if per_type:
        scaled_diffs = {}
        for name, val in var_types.items():
            if len(val)>0:
                scaled_diffs[name] = np.nanmean(result[:,val], axis=1)
    else:
        scaled_diffs = result

    return scaled_diffs

def get_scaled_error(sigma_imp, sigma):
    """
    gets a scaled error between matrices |simga - sigma_imp|_F^2 / |sigma|_F^2
    """
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)

def grassman_dist(A,B):
    U1, d1, _ = np.linalg.svd(A, full_matrices = False)
    U2, d2, _ = np.linalg.svd(B, full_matrices = False)
    d = svdvals(np.dot(U1.T, U2))
    theta = np.arccos(d)
    return np.linalg.norm(theta), np.linalg.norm(d1-d2)

def error_by_reliability(error, r, xtrue, ximp, num=100, start=1):
    q = np.linspace(0, 1-start/num, num=num)
    r_q = np.nanquantile(r, q)
    err = np.zeros(num)
    loc_missing = ~np.isnan(r)
    r, xtrue, ximp = r[loc_missing], xtrue[loc_missing], ximp[loc_missing]
    for i,x in enumerate(r_q):
        #  keep entries with top reliabilities 
        loc_q = r >= x

        val, imp = xtrue[loc_q], ximp[loc_q]
        if error == 'NRMSE':
            err[i] = np.sqrt(np.power(val-imp, 2).mean()) / np.sqrt(np.power(val, 2).mean()) 
        elif error == 'RMSE':
            err[i] = np.sqrt(np.power(val-imp, 2).mean()) 
        elif error == 'MAE':
            err[i] = np.abs(val-imp).mean()
        else: 
            raise ValueError('Error can only be one of NRMSE, RMSE, MAE.')
    return err




