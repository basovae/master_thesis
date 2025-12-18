import copy

import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


class CVTArchive:
    def __init__(self, n_niches, behavior_dim, cvt_samples=25000):
        """
        Args:
            n_niches: Number of cells (paper uses 1024-10000)
            behavior_dim: Dimensionality of behavior descriptor
            cvt_samples: Samples for centroid initialization
        """

        samples = np.random.rand(cvt_samples, behavior_dim)
    
    # Official uses KMeans class, not k_means function
    
        k_means = KMeans(init='k-means++', n_clusters=n_niches,
                     n_init=1, max_iter=1000000, verbose=1, tol=1e-8)
        k_means.fit(samples)
        self.centroids = k_means.cluster_centers_
    
        self._kd_tree = KDTree(self.centroids, leaf_size=30, metric='euclidean')

        # Archive storage
        self.fitnesses = np.full(n_niches, -np.inf)
        self.policies = [None] * n_niches
        self.behavior_descriptors = [None] * n_niches

    def get_cell_index(self, behavior_descriptor):
        """Find nearest centroid (Algorithm 1, line: get cell index)"""
        _, index = self._kd_tree.query(behavior_descriptor)
        return index

    def add_to_archive(self, policy, fitness, behavior_descriptor):
        """Elite replacement (Algorithm 1, lines 22-23)"""
        cell_idx = self.get_cell_index(behavior_descriptor)

        # Replace only if higher fitness OR cell empty
        if self.fitnesses[cell_idx] < fitness:
            self.fitnesses[cell_idx] = fitness
            self.policies[cell_idx] = copy.deepcopy(policy)
            self.behavior_descriptors[cell_idx] = behavior_descriptor
            return True, cell_idx
        return False, cell_idx

    def uniform_selection(self, n=1):
        """Sample n policies uniformly from occupied cells"""
        occupied = [i for i in range(len(self.policies)) if self.policies[i] is not None]
        if len(occupied) == 0:
            return []
        indices = np.random.choice(occupied, size=min(n, len(occupied)), replace=True)
        return [self.policies[i] for i in indices]

    # Example: 2D behavior space for portfolio optimization
    def compute_behavior_descriptor(portfolio_weights, returns_history):
        """
        Extract behavioral characteristics from portfolio execution.

        Returns:
            bd: [volatility, diversification] or other 2D+ descriptor
        """
        # Volatility (realized)
        portfolio_returns = returns_history @ portfolio_weights
        volatility = np.std(portfolio_returns)

        # Diversification (inverse Herfindahl)
        herfindahl = np.sum(portfolio_weights ** 2)
        diversification = 1.0 - herfindahl  # Higher = more diversified

        return np.array([volatility, diversification])