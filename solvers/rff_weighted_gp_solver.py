import numpy as np

from solvers.settings import RffWeightedGPExperimentSetting
from solvers.solver import Solver


class RffWeightedGPSolver(Solver):
    def __init__(self, number_of_dim, samples, settings: RffWeightedGPExperimentSetting):
        super().__init__(number_of_dim, samples, settings)
        self.rff_dim = settings.RFF_DIM
        self.alpha_ = None
        self.lambda_ = settings.LAMBDA
        self.b = settings.RFF_B
        self.W = settings.RFF_W
        self.number_of_dim = number_of_dim
        self.sigma_2 = 0.5

        self.ls = np.array([settings.LS] * number_of_dim)
        self.phi_normalized = settings.PHI_NORMALIZED
        self.sigma = 1
        self.beta = settings.BETA

        self.mu = None
        self.std = None

        self.problem = []

        # TODO maximum inner product searcnh

    def _fit(self):
        self.mu, self.std = self._get_posterior(self.X_grid)

    def best_point(self):
        id_max = np.argmax(self.mu + self.beta * self.std)
        new_we = self._get_new_weight(self.X_grid[id_max])
        self.weights = np.append(self.weights, new_we)
        return self.X_grid[id_max]

    def _get_phi(self, X, weights=True):
        if self.phi_normalized:
            return self._get_phi_normalized(X, weights)
        else:
            return self._get_phi_unnormalized(X, weights)

    def _get_phi_normalized(self, X, weights=True):
        # print("aaaa")
        phi = np.zeros((X.shape[0], self.rff_dim))
        for i, x in enumerate(X):
            features = self._map_point_to_rff(x.reshape(1, -1))
            features = np.squeeze(features)
            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(self.sigma_2) * features
            if weights:
                features = features * (1 / self.weights[i])
            phi[i] = features
        return phi

    def _get_phi_unnormalized(self, X, weights=True):
        phi = self._map_point_to_rff(X)
        phi = np.sqrt(self.sigma_2) * phi
        if weights:
            phi = phi / self.weights
        return phi.T

    def _get_simga_inv(self, phi):
        return np.linalg.inv(phi.T @ phi + self.lambda_ * np.eye(self.rff_dim))

    def _get_y_weighted(self, Y):
        diag = np.diag(1 / self.weights)
        return np.matmul(diag, Y.reshape(-1, 1))

    def _get_prior(self, X, Y):
        phi = self._get_phi(X)
        sigma_inv = self._get_simga_inv(phi)
        y_weighted = self._get_y_weighted(Y)
        return sigma_inv @ phi.T @ y_weighted, sigma_inv

    def _get_new_weight(self, x):
        x = self._get_phi(x.reshape(1, -1), weights=False)
        sigma_inv = self._get_simga_inv(x)
        var = self.lambda_ * np.squeeze(np.dot(np.dot(x, sigma_inv), x.T))
        return np.sqrt(var) / np.sqrt(self.lambda_)

    def _get_posterior(self, X):
        prior_mu, sigma_inv = self._get_prior(np.array(self.X), np.array(self.y))
        phi = self._get_phi(X, weights=False)

        mean = np.squeeze(phi @ prior_mu)
        std = np.einsum('ij,jk,ki->i', phi, sigma_inv, phi.T)
        std = np.sqrt(self.lambda_ * std)
        return mean, std

    def _map_point_to_rff(self, x):
        """
        X: np.array of shape (n, d)
        """
        if self.W is None:
            self._generate_rff_vectors()

        B = np.repeat(self.b[:, np.newaxis], x.shape[0], axis=1)
        Z = np.sqrt(2 / self.rff_dim) * np.cos(self.sigma * self.W @ x.T + B)
        return Z

    def _generate_rff_vectors(self):
        if self.W is None:
            self.W = np.random.multivariate_normal(np.zeros(self.number_of_dim),
                                                   1 / (self.ls ** 2) * np.identity(self.number_of_dim), self.rff_dim)
            self.b = np.random.uniform(0, 2 * np.pi, size=self.rff_dim)
