import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, poisson

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=True):
    """ 
    Very close to numpy.percentile, but supports weights. NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.

    Acknowledgement: code from Alleo's answer in stackoverflow 
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def inverse_ecdf(x, x_obs, DECIMAL_PRECISION = 3):
    """
    computes the inverse ecdf (quantile) for x with ecdf given by data
    """
    data = x_obs
    n = len(data)
    if n==0:
        print('No observation can be used for imputation')
        raise
    # round to avoid numerical errors in ceiling function
    quantile_indices = np.ceil(np.round_((n + 1) * x - 1, DECIMAL_PRECISION))
    quantile_indices = np.clip(quantile_indices, a_min=0,a_max=n-1).astype(int)
    sort = np.sort(data)
    return sort[quantile_indices]


def truncated_marginal_lower(x, x_obs):
    xmin = x_obs.min()
    loc = np.isclose(x_obs, xmin)
    q_lower = loc.mean()
    func = ECDF(x_obs[~loc])

    # from scores to lower & upper bounds
    lower = np.empty_like(x)
    upper = np.empty_like(x)
    loc_x = np.isclose(x, xmin)
    lower[loc_x] = -np.inf
    upper[loc_x] = norm.ppf(q_lower)
    # Put same values at non truncated entries for identification purpose from truncated entries
    # avoid 1 in scores, no need to avoid 0 since it is at least q_lower
    n = loc.sum()
    loc_x_nontrun = ~loc_x
    q_nontrun = q_lower + (1-q_lower) * func(x[loc_x_nontrun]) * n/(n+1)
    lower[loc_x_nontrun] = norm.ppf(q_nontrun)
    upper[loc_x_nontrun] = lower[loc_x_nontrun].copy()
    return lower, upper

def truncated_marginal_upper(x, x_obs):
    xmax = x_obs.max()
    loc = np.isclose(x_obs, xmax)
    q_upper = loc.mean()
    func = ECDF(x_obs[~loc])

    # from scores to lower & upper bounds
    lower = np.empty_like(x)
    upper = np.empty_like(x)
    loc_x = np.isclose(x, xmax)
    lower[loc_x] = norm.ppf(1-q_upper)
    upper[loc_x] = np.inf
    # Put same values at non truncated entries for identification purpose from truncated entries
    # avoid 0 in scores, no need to avoid 1 since it is at most 1-q_upper
    n = loc.sum()
    loc_x_nontrun = ~loc_x
    q_nontrun = (1-q_upper) * func(x[loc_x_nontrun])
    q_nontrun[q_nontrun == 0] = 1/(2*(n+1)) 
    lower[loc_x_nontrun] = norm.ppf(q_nontrun)
    upper[loc_x_nontrun] = lower[loc_x_nontrun].copy()
    return lower, upper

def truncated_marginal_twoside(x, x_obs):
    xmin = x_obs.min()
    xmax = x_obs.max()
    loc_upper = np.isclose(x_obs, xmax)
    loc_lower = np.isclose(x_obs, xmin)
    q_upper = loc_upper.mean()
    q_lower = loc_lower.mean()
    loc_nontrun = ~(loc_upper | loc_lower)
    func = ECDF(x_obs[loc_nontrun])

    # from scores to lower & upper bounds
    lower = np.empty_like(x)
    upper = np.empty_like(x)
    loc_x_upper = np.isclose(x, xmax)
    loc_x_lower = np.isclose(x, xmin)
    lower[loc_x_lower] = -np.inf
    upper[loc_x_lower] = norm.ppf(q_lower)
    lower[loc_x_upper] = norm.ppf(1-q_upper)
    upper[loc_x_upper] = np.inf
    # Put same values at non truncated entries for identification purpose from truncated entries
    # no need to avoid 0 or 1 for scores at non truncated entries
    # the values range from q_lower to 1-q_upper
    loc_x_nontrun = ~(loc_x_upper | loc_x_lower)
    q_nontrun = q_lower + (1-q_lower-q_upper) * func(x[loc_x_nontrun])
    lower[loc_x_nontrun] = norm.ppf(q_nontrun)
    upper[loc_x_nontrun] = lower[loc_x_nontrun].copy()
    return lower, upper

def truncated_inverse_marginal_lower(q, x_obs, eps=1e-6):
    x = x_obs
    xmin = x.min()
    loc_lower = np.isclose(x, xmin)
    loc_nontrun = ~loc_lower
    
    q_lower = loc_lower.mean()
    x_nontrun = x[loc_nontrun]
    
    x_imp = np.empty_like(q)
    imp_lower = q<=q_lower+eps
    x_imp[imp_lower] = xmin
    imp_nontrun = ~imp_lower
    q_adjusted = (q[imp_nontrun]-q_lower)/(1-q_lower)
    x_imp[imp_nontrun] = np.quantile(x_nontrun, q_adjusted)
    
    return x_imp
    
def truncated_inverse_marginal_upper(q, x_obs, eps=1e-6):
    x = x_obs
    xmax = x.max()
    loc_upper = np.isclose(x, xmax)
    loc_nontrun = ~loc_upper
    
    q_upper = loc_upper.mean()
    x_nontrun = x[loc_nontrun]
        
    x_imp = np.empty_like(q)
    imp_upper = q>=1-q_upper-eps
    x_imp[imp_upper] = xmax
    imp_nontrun = ~imp_upper
    q_adjusted = q[imp_nontrun]/(1-q_upper)
    x_imp[imp_nontrun] = np.quantile(x_nontrun, q_adjusted)
    
    return x_imp

def truncated_inverse_marginal_twoside(q, x_obs, eps=1e-6):
    x = x_obs
    xmax = x.max()
    xmin = x.min()
    loc_upper = np.isclose(x, xmax)
    loc_lower = np.isclose(x, xmin)
    loc_nontrun = ~(loc_upper | loc_lower)
    
    q_upper = loc_upper.mean()
    q_lower = loc_lower.mean()
    x_nontrun = x[loc_nontrun]
        
    x_imp = np.empty_like(q)
    imp_upper = q>=1-q_upper-eps
    imp_lower = q<=q_lower+eps
    imp_nontrun = ~(imp_upper|imp_lower)
    x_imp[imp_upper] = xmax
    x_imp[imp_lower] = xmin
    q_adjusted = (q[imp_nontrun]-q_lower)/(1-q_upper-q_lower)
    x_imp[imp_nontrun] = np.quantile(x_nontrun, q_adjusted)
    
    return x_imp