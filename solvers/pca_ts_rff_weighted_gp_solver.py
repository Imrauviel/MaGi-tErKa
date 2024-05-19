import numpy as np
from sklearn.decomposition import PCA

from solvers import TSRffWeightedGPSolver


class PcaTSRffWeightedGPSolver(TSRffWeightedGPSolver):
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
        prior_mu, self.sigma_inv = self._get_prior(self.X_transformed, np.array(self.y))

        try:
            prob = np.random.multivariate_normal(prior_mu.flatten(), self.sigma_inv)
        except np.linalg.LinAlgError:
            print(np.array(self.X), np.array(self.y), prior_mu, self.sigma_inv)
            prob = np.random.multivariate_normal(prior_mu.flatten(), self.sigma_inv + 1e-6 * np.eye(self.sigma_inv.shape[0]))

        self.phi = self._get_phi(X, weights=False)

        mean = np.squeeze(self.phi @ prob)

        std = np.einsum('ij,jk,ki->i', self.phi, self.sigma_inv, self.phi.T)
        std = np.clip(std, a_min=0, a_max=None)
        std = np.sqrt(self.lambda_ * std)

        return mean, std
