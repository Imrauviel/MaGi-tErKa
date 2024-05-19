import numpy as np

from solvers.ucb_gp_solver import UcbGPSolver


class TSGPSolver(UcbGPSolver):
    def __init__(self, number_od_dim, samples, settings):
        super().__init__(number_od_dim, samples, settings)

    def best_point(self):
        values = np.random.normal(self.mu, self.sigma)
        max_idx = np.argmax(values)
        return self.X_grid[max_idx]
