import numpy as np
import matplotlib.pyplot as plt
class OnlineOrdinalMarginalEstimator():
    def __init__(self, X=None):
        self.ord_counts = {}
        self.num_distinct_ords = 0
        self.num_total_ords = 0
        if X is not None:
            self.partial_fit(X)

    def partial_fit(self, X_batch):
        """
        Update the marginal estimate with the data in X_batch
        """
        for x in X_batch:
            if x in self.ord_counts:
                self.ord_counts[x] += 1
            else:
                self.ord_counts[x] = 1
                self.num_distinct_ords += 1
            self.num_total_ords += 1

    def get_cdf(self, X_batch):
        """
        get the cdf at each point in X_batch
        """
        # the lower endpoint of the interval for the cdf
        cdf_lower = np.zeros(X_batch.shape).astype(float)
        # the upper endpoint of the interval for the cdf
        cdf_upper = np.zeros(X_batch.shape).astype(float)
        for threshold in self.ord_counts:
            threshold_count = self.ord_counts[threshold]
            cdf_lower += ((threshold < X_batch)*threshold_count)/self.num_total_ords
            cdf_upper +=  ((threshold <= X_batch)*threshold_count)/self.num_total_ords
        return cdf_lower,cdf_upper


    def get_inverse_cdf(self, Q_batch):
        """
        Gets the inverse CDF of Q_batch
        returns: the Q_batch quantiles of the ordinals seen thus far
        """
        q = 0.0
        quantiles = np.empty(Q_batch.shape)
        for i,threshold in enumerate(sorted(self.ord_counts)):
            if i == 0:
                quantiles[:] = threshold
            count = self.ord_counts[threshold]
            quantiles[q < Q_batch] = threshold
            q += count/float(self.num_total_ords)
        return quantiles

    def get_density(self, X_batch):
        """
        Get the density at each point in X_batch
        """
        densities = np.zeros(X_batch.shape)
        for i,x in enumerate(X_batch):
            if x in self.ord_counts:
                densities[i] = self.ord_counts[x]/self.num_total_ords
        return densities

        