import numpy as np

from solvers.settings import RffExperimentSetting
from solvers.solver import Solver
from scipy.linalg import cholesky, cho_solve


class RFFSolver(Solver):

    def __init__(self, number_od_dim, samples, settings: RffExperimentSetting):
        """Gaussian process regression using random Fourier features.
        """

        super().__init__(number_od_dim, samples, settings)

        self.rff_dim = settings.RFF_DIM
        self.sigma = settings.SIGMA
        self.alpha_ = None
        self.b_ = None
        self.W_ = None

        self.y_mean = None
        self.y_std = None

        self.X_grid_sample = None

    def _fit(self):
        X_pred = np.array(self.X)

        N, _ = X_pred.shape
        Z, W, b = self._get_rffs(X_pred, return_vars=True)
        sigma_I = self.sigma * np.eye(N)
        self.kernel_ = Z.T @ Z + sigma_I

        # Solve for Rasmussen and William's alpha.
        lower = True
        L = cholesky(self.kernel_, lower=lower)
        self.alpha_ = cho_solve((L, lower), self.y)

        # Save for `predict` function.
        self.Z_train_ = Z
        self.L_ = L
        self.b_ = b
        self.W_ = W

        if self.num_points > 10 ** 4:
            self.X_grid_sample = self.X_grid[np.random.choice(self.X_grid.shape[0], 10 ** 4, replace=False), :]
        else:
            self.X_grid_sample = self.X_grid

        Z_test = self._get_rffs(self.X_grid_sample, return_vars=False)
        K_star = Z_test.T @ self.Z_train_
        self.y_mean = K_star.dot(self.alpha_)

        lower = True
        v = cho_solve((self.L_, lower), K_star.T)
        y_cov = (Z_test.T @ Z_test) - K_star.dot(v)
        self.y_std = np.sqrt(np.diag(y_cov))
        # self.y_std_approx = np.full(self.y_mean.shape, np.sqrt(self.sigma)) #TODO: co się dzieje, kiedy użyjemy OG std

        return self

    def best_point(self):
        max_idx = np.argmax(self.y_mean + self.y_std)  # TODO: param beta
        # print(max_idx, np.argmax(self.y_mean))
        return self.X_grid_sample[max_idx]

    def _get_rffs(self, X, return_vars):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
            b = np.random.uniform(0, 2 * np.pi, size=self.rff_dim)

        B = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1. / np.sqrt(self.rff_dim)
        Z = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)
        if return_vars:
            return Z, W, b
        return Z

    def _get_rvs(self, D):
        """On first call, return random variables W and b. Else, return cached
        values.
        """
        if self.W_ is not None:
            return self.W_, self.b_
        W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
        b = np.random.uniform(0, 2 * np.pi, size=self.rff_dim)
        return W, b
