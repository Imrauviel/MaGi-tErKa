import abc
import time

import numpy as np

from solvers.settings import ExperimentSetting


class Solver(abc.ABC):
    def __init__(self, number_od_dim, samples, settings: ExperimentSetting):

        if settings.X_GRID is not None:
            self.X_grid = settings.X_GRID
        else:
            num_points = int(10 ** (settings.CONSTANT_POWER / number_od_dim))
            ranges = [np.linspace(0, 1, num_points) for _ in range(number_od_dim)]
            self.meshgrid = np.array(np.meshgrid(*ranges))
            self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T

        self.X, self.y = samples[0], samples[1]
        self.fit_time_history = []
        self.num_points = self.X_grid.shape[0]
        self.weights = np.array([1.0] * len(self.X))

    def fit(self):
        start_time = time.time()
        self._fit()
        self.fit_time_history.append(time.time() - start_time)

    @abc.abstractmethod
    def _fit(self):
        ...

    @abc.abstractmethod
    def best_point(self):
        ...
