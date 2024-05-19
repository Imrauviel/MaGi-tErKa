import numpy as np
from sklearn.decomposition import PCA

from solvers import RffWeightedGPSolver


class PcaRffWeightedGPSolver(RffWeightedGPSolver):
    def __init__(self, number_of_dim, samples, settings):
        super().__init__(2, samples, settings)
        self.pca = None
        self.X_transformed = None
        self.pca = PCA(n_components=2)
        self.x_grid_transformed = self.pca.fit_transform(self.X_grid)

    def _fit(self):
        self.X_transformed = self.pca.transform(np.array(self.X))
        self.mu, self.std = self._get_posterior(self.x_grid_transformed)

    def _get_posterior(self, X):

        prior_mu, sigma_inv = self._get_prior(self.X_transformed, np.array(self.y))
        phi = self._get_phi(X, weights=False)

        mean = np.squeeze(phi @ prior_mu)
        std = np.einsum('ij,jk,ki->i', phi, sigma_inv, phi.T)
        std = np.sqrt(self.lambda_ * std)
        return mean, std
