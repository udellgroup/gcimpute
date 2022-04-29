import numpy as np
from scipy.stats import norm
import warnings

a1 =  0.254829592
a2 = -0.284496736
a3 =  1.421413741
a4 = -1.453152027
a5 =  1.061405429
erf_p  =  0.3275911

def erf(x):
    # save the sign of x
    sign = 2 * (x>=0) -1 
    x = np.abs(x)

    # A&S formula 7.1.26
    t = 1/(1+erf_p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def norm_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return np.exp(-np.power(x,2)/2) / np.sqrt(2 * np.pi)

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

def get_truncnorm_moments_vec(a,b,mu,std,tol=1e-6, mean_only=False, warn_tol=-1e-4):
    a,b,mu,std = np.array(a), np.array(b), np.array(mu), np.array(std)
    alpha, beta = (a-mu)/std, (b-mu)/std

    Z = norm_cdf(beta) - norm_cdf(alpha)
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

    pdf_beta, pdf_alpha = norm_pdf(beta_work), norm_pdf(alpha_work)
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
                s = 1 - r2_dict[name] - (r1[loc]**2)
                if s.min()<0:
                    if s.min()<warn_tol:
                        m = f'Negative turnc std: {s.min()}'
                        warnings.warn(m)
                    s[s<0] = 0
                _std[abs_loc] = std[abs_loc] * np.sqrt(s)
        _std[fail_loc] = np.inf
        _std[trivial_loc] = std[trivial_loc]

        out['std'] = _std
    return out