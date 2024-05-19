import numpy as np

from solvers import RffWeightedGPSolver


class TSRffWeightedGPSolver(RffWeightedGPSolver):
    def __init__(self, number_of_dim, samples, settings):
        super().__init__(number_of_dim, samples, settings)

        self.phi = None
        self.sigma_inv = None

        self.d = 0

    def _get_posterior(self, X):
        prior_mu, sigma_inv = self._get_prior(np.array(self.X), np.array(self.y))
        # print("A", prior_mu.shape, sigma_inv.shape)
        try:
            prob = np.random.multivariate_normal(prior_mu.flatten(), sigma_inv)
        except np.linalg.LinAlgError:
            print(np.array(self.X), np.array(self.y), prior_mu, sigma_inv)
            prob = np.random.multivariate_normal(prior_mu.flatten(), sigma_inv + 1e-6 * np.eye(sigma_inv.shape[0]))
        phi = self._get_phi(X, weights=False)

        mean = np.squeeze(phi @ prob)
        # print("B", mean.shape)
        std = np.einsum('ij,jk,ki->i', phi, sigma_inv, phi.T)
        std = np.clip(std, a_min=0, a_max=None)
        self.phi = phi  # TODO: clean up
        self.sigma_inv = sigma_inv
        std = np.sqrt(self.lambda_ * std)
        return mean, std

    def _fit(self):
        self.mu, self.std = self._get_posterior(self.X_grid)

    def best_point(self):
        self.d += 1
        # if self.d % 100 == 0:
        id_max = np.argmax(self.mu)
        # new_we = self.std[id_max] / np.sqrt(self.lambda_)
        new_we = self._get_new_weight(self.X_grid[id_max])
        if not np.isnan(new_we) or new_we < 0.000000001 or new_we != 0.0:
            self.weights = np.append(self.weights, new_we)
        else:
            # print("NAN", self.std[id_max])
            self.problem.append((self.phi, self.sigma_inv))
            self.weights = np.append(self.weights, 0.00000001)
        return self.X_grid[id_max]
