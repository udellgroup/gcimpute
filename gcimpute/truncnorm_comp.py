import numpy as np
from scipy.stats import norm



def get_truncnorm_moments(a,b,mu,std,tol=1e-6, mean_only=False):
    alpha, beta = (a-mu)/std, (b-mu)/std
    Z = norm.cdf(beta) - norm.cdf(alpha)
    assert np.isfinite(Z), f'Z is {Z}'
    if Z < tol:
        return np.inf, np.inf
    pdf_beta, pdf_alpha = norm.pdf(beta), norm.pdf(alpha)
    assert np.isfinite(pdf_alpha), f'pdf_alpha is {pdf_alpha}'
    assert np.isfinite(pdf_beta), f'pdf_beta is {pdf_beta}'
    r1 = (pdf_beta - pdf_alpha) / Z
    _mean = mu - r1 * std
    if mean_only:
        return _mean
    if beta >= np.inf:
        assert np.isfinite(alpha), f'alpha is {alpha} when beta is inf'
        r2 = (-alpha * pdf_alpha) / Z
    elif alpha <= -np.inf:
        assert np.isfinite(beta), f'beta is {beta} when alpha is -inf'
        r2 = (beta * pdf_beta) / Z
    else:
        r2 = (beta * pdf_beta - alpha * pdf_alpha) / Z
    _std = std * np.sqrt(1 - r2 - (r1**2)) 
    return _mean, _std

def get_truncnorm_moments_vec(a,b,mu,std,tol=1e-6, mean_only=False):
    a,b,mu,std = np.array(a), np.array(b), np.array(mu), np.array(std)
    alpha, beta = (a-mu)/std, (b-mu)/std

    Z = norm.cdf(beta) - norm.cdf(alpha)
    assert np.isfinite(Z).all() and Z.min()>=0, f'Z is {Z}'
    work_loc = np.flatnonzero((Z>tol) & (Z<1))
    trivial_loc = Z==1
    fail_loc = Z<=tol
    Z_work = Z[work_loc]
    mu_work = mu[work_loc]
    sigma_work = std[work_loc]
    beta_work, alpha_work = beta[work_loc], alpha[work_loc]

    _mean = np.zeros_like(mu)
    _std = np.zeros_like(mu)

    out = {}

    pdf_beta, pdf_alpha = norm.pdf(beta_work), norm.pdf(alpha_work)
    assert np.isfinite(pdf_alpha).all(), f'pdf_alpha is {pdf_alpha}'
    assert np.isfinite(pdf_beta).all(), f'pdf_beta is {pdf_beta}'
    r1 = (pdf_beta - pdf_alpha) / Z_work
    _mean[work_loc] = mu_work - r1 * sigma_work
    _mean[fail_loc] = np.inf
    _mean[trivial_loc] = mu[trivial_loc]
    out['mean'] = _mean

    # 
    if not mean_only:
        loc_dict = {}
        r2_dict = {}

        loc = beta_work >= np.inf
        if any(loc):
            assert np.isfinite(alpha_work[loc]).any(), f'alpha is {alpha_work[loc]} when beta is inf'
            r2_dict['inf_beta'] = (-alpha_work[loc] * pdf_alpha[loc]) / Z_work[loc]
            loc_dict['inf_beta'] = loc

        loc = alpha_work <= -np.inf
        if any(loc):
            assert np.isfinite(beta_work[loc]).any(), f'alpha is {beta_work[loc]} when alpha is -inf'
            r2_dict['inf_alpha'] = (beta_work[loc] * pdf_beta[loc]) / Z_work[loc]
            loc_dict['inf_alpha'] = loc

        loc = (beta_work < np.inf) & (alpha_work > -np.inf)
        if any(loc):
            r2_dict['finite'] = (beta_work[loc] * pdf_beta[loc] - alpha_work[loc] * pdf_alpha[loc]) / Z_work[loc]
            loc_dict['finite'] = loc

        for name, loc in loc_dict.items():
            abs_loc = work_loc[loc]
            if any(loc):
                _std[abs_loc] = std[abs_loc] * np.sqrt(1 - r2_dict[name] - (r1[loc]**2))
        _std[fail_loc] = np.inf
        _std[trivial_loc] = std[trivial_loc]

        out['std'] = _std
    return out