from abc import ABC, abstractmethod


class Problem(ABC):
    @abstractmethod
    def reward_function(self, param1, param2, param3, param4, param5, param6):
        pass

    @abstractmethod
    def get_bounds(self):
        pass