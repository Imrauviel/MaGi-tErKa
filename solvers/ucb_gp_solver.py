import numpy as np

from solvers.settings import UCBExperimentSetting
from solvers.solver import Solver
from sklearn.gaussian_process import kernels, GaussianProcessRegressor


class UcbGPSolver(Solver):
    def __init__(self, number_od_dim, samples, settings: UCBExperimentSetting):
        super().__init__(number_od_dim, samples, settings)
        self.gp = None
        self.mu = None
        self.sigma = None
        self.beta = settings.BETA

    def fit_gp(self, X, y):
        k_cov = kernels.RBF(0.1, (0.01, 10))
        k_scaling = kernels.ConstantKernel(1, (0.1, 10))
        k_noise = kernels.WhiteKernel(noise_level=np.ptp(y) / 10, noise_level_bounds=(1e-6, 1e-3))
        self.gp = GaussianProcessRegressor(
            kernel=k_scaling * k_cov + k_noise,
            random_state=0
        ).fit(X, y)

    def _fit(self):
        self.fit_gp(self.X, self.y)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)

    def best_point(self):
        max_idx = np.argmax(self.mu + self.sigma * np.sqrt(self.beta))
        return self.X_grid[max_idx]
