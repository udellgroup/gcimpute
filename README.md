# gcimpute
This package provides a python implemention to impute missing values by fitting a Gaussian copula model, on incomplete dataset thay may contain continuous, ordinal, binary and truncated variables. The user could either fit a full rank Gaussian copula model [1] or a low rank Gaussian copula model [2]. The package also includes the mini-batch and online implementation of the full rank Gaussian copula model [3], which can accelerate the training process and adapt to a changing distribution in the streaming data. The fitted model can also be used for latent correlation learning and latent correlation change point detection (only for online data).

## Installation

The easiest way is to install using pip: 

`
pip install gcimpute
` 

If you want to customize the source code, you may install in the editable mode by first `git clone` this respository, and then do

`
pip install -e .
`

in the cloned directory.


## Examples 
```python
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_data import generate_mixed_from_gc
from gcimpute.helper_evaluation import get_smae
from gcimpute.helper_mask import mask_MCAR
import numpy as np

# generate and mask 15-dim mixed data (5 continuous variables, 5 ordinal variables (1-5) and 5 boolean variables) 
X = generate_mixed_from_gc(n=2000)
X_mask = mask_MCAR(X, mask_fraction=0.4)

# model fitting 
model = GaussianCopula(verbose=1)
X_imp = model.fit_transform(X=X_mask)

# Evaluation: compute the scaled-MAE (SMAE) for each data type (scaled by MAE of median imputation) 
smae = get_smae(X_imp, X, X_mask)
print(f'The SMAE across continous variables: mean {smae[:5].mean():.3f} and std {smae[:5].std():.3f}')
print(f'The SMAE across oridnal variables: mean {smae[5:10].mean():.3f} and std {smae[5:10].std():.3f}')
print(f'The SMAE across boolean variables: mean {smae[10:].mean():.3f} and std {smae[10:].std():.3f}')
```
More detailed examples are available under directory Examples. Especially, the [main tutorial](https://github.com/udellgroup/gcimpute/blob/master/Examples/Main_Tutorial.ipynb) covers most functions' usage, and thus is a must-read if you are using our software for the first time.

## News
Our software went through a few name and structural changes. It was previously named online_gc_imp, and then GaussianCopulaImp. The current name gcimpute is our final pick. It is very unlikely we will further change its name. The structural changes are substantial: we improved the code quality and the user interface. It now has an interface consistent with `sklearn.impute`. If you are an early user of our software, you may find your codes using our software no longer works if using our new release. We greatly appreciate your interest in our work and sincerely apologize for any inconvenience you may experience. The [notebook](https://github.com/udellgroup/gcimpute/blob/master/Examples/Main_Tutorial.ipynb) will help you quickly modify your codes using our current release.

## References
[1] Zhao, Y. and Udell, M. Missing value imputation for mixed data via Gaussian copula, KDD 2020.

[2] Zhao, Y. and Udell, M. Matrix completion with quantified uncertainty through low rank Gaussian copula, NeurIPS 2020.

[3] Zhao, Y., Landgrebe, E., Shekhtman E., and Udell, M. Online Missing Value Imputation and Change Point Detection
with the Gaussian Copula, AAAI 2022
