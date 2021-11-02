import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from .transform_function import TransformFunction


class OnlineTransformFunction(TransformFunction):
    def __init__(self, cont_indices, ord_indices, window_size=100):
        """
        Require window_size to be positive integers.

        To initialize the window, 
        for continuous columns, sample standatd normal with mean and variance determined by the first batch of observation;
        for ordinal columns, sample uniformly with replacement among all integers between min and max seen from data.

        """
        p = len(cont_indices)
        window_init = np.ones((window_size, p), dtype=np.float64) * np.nan
        # window is stored in self.X
        super().__init__(X=window_init, cont_indices=cont_indices, ord_indices=ord_indices)
        self.window_size = window_size
        self.update_pos = np.zeros(p, dtype=np.int64)


    def init_window(self, X_batch):
        """
        Initialize the running window for marginal estimate. The initialized values will be forgot gradually.
        """
        # Continuous columns: normal initialization
        if any(self.cont_indices):
            mean_cont = np.nanmean(X_batch[:, self.cont_indices])
            std_cont = np.nanstd(X_batch[:, self.cont_indices])
            mean_init = 0 if np.isnan(mean_cont) else mean_cont
            std_init = 1 if np.isnan(std_cont) else std_cont
            self.X[:, self.cont_indices] = np.random.normal(mean_init, std_init, size=(self.window_size, np.sum(self.cont_indices)))
        # Ordinal columns: uniform initialization
        for j,loc in enumerate(self.ord_indices):
            if loc:
                min_ord = np.nanmin(X_batch[:,j])
                max_ord = np.nanmax(X_batch[:,j]) 
                if np.isnan(min_ord):
                    self.X[:,j].fill(0)
                else:
                    self.X[:,j] = np.random.randint(min_ord, max_ord+1, size=self.window_size)
        
    def update_window(self, X_batch):
        """
        Update the running window used to estimate marginals with the data in X
        """
        if np.isnan(self.X[0,0]):
            self.init_window(X_batch)

        # update for new data
        for j, x_col in enumerate(X_batch.T):
            obs_indices = ~np.isnan(x_col)
            x_obs = x_col[obs_indices]
            num_obs = len(x_obs)
            if num_obs>0:
                old_pos = self.update_pos[j]
                if num_obs<self.window_size:
                    new_pos = (old_pos+num_obs)%self.window_size
                    if old_pos < new_pos:
                        indices = np.arange(old_pos, new_pos, 1)
                    else:
                        _part_1 = np.arange(old_pos, self.window_size, 1)
                        _part_2 = np.arange(new_pos)
                        indices = np.concatenate((_part_1, _part_2))
                    self.X[indices, j] = x_obs
                    self.update_pos[j] = new_pos
                else:
                    # update the whole window
                    _part_1 = np.arange(old_pos, self.window_size, 1)
                    _part_2 = np.arange(old_pos)
                    indices = np.concatenate((_part_1, _part_2))
                    # update the window using most recent observation
                    # update pos does not change 
                    self.X[indices, j] = x_obs[-self.window_size:]





    
