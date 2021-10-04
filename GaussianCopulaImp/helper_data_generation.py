import numpy as np 
from scipy.stats import random_correlation, norm, expon

def _cont_to_ord(x, k, by = 'dist', seed=1, q_min=0.05, q_max=0.95):
    """
    convert entries of x to an ordinal with k levels (0, 1, ..., k-1) using thresholds selected by one choice of the following:
    by = 'dist': select evenly spaced thresholds
    by = 'quantile': select random observations between .05 quantile and .95 quantile as thresholds
    """
    # make the cutoffs based on the quantiles
    np.random.seed(seed)
    std_dev = np.std(x)
    if by == 'dist':
        # length of k-1 
        # the head and the tail are removed because no value is smaller than the head and 
        # no value is larger than the tail
        cutoffs = np.linspace(np.min(x), np.max(x), k+1)[1:-1]
    elif by == 'quantile':
        # samping cutoffs from 5% quantile to 95% quantile 
        select = (x>np.quantile(x, q_min)) & (x<np.quantile(x, q_max))
        if sum(select)<k:
            raise ValueError(f'Cannot ordinalize the variable because there are fewer than {k} observations from .05 percentile to .95 percentile')
        cutoffs = np.random.choice(x[select], k-1, replace = False)
    else:
        raise ValueError('Unsupported cutoff_by option')
    ords = np.digitize(x, cutoffs)
    return ords


def cont_to_ord(x, k, by = 'dist', seed=1, max_try=20):
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
        if c==max_try:
            raise ValueError("Failure: the ordinalized variable always has fewer than {k} levels in {max_try} attempts")
    return result

def generate_sigma(seed, p=15):
    '''
    Generate a random correlation matrix 
    '''
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