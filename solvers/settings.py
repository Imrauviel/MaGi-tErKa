import numpy as np
from pydantic_settings import BaseSettings


# TODO: Zrobić porządek: solver do osbonych plików i dodanie do nich tego settingu

class ExperimentSetting(BaseSettings):
    CONSTANT_POWER: int = 4
    QUANTUM_NOISE: bool = False
    INIT_POINTS: int = 1
    VARIANCE: float = 0.01 # 0.05
    X_GRID: np.ndarray[float] | None = None
    NUMBER_OF_QUBITS: int = 6


class RffWeightedGPExperimentSetting(ExperimentSetting):
    RFF_DIM: int = 200
    LS: float = 0.2
    PHI_NORMALIZED: bool = False
    BETA: float = 1.0
    LAMBDA: float = 1.0
    RFF_W: np.ndarray[float] | None = None
    RFF_B: np.ndarray[float] | None = None


class RffExperimentSetting(ExperimentSetting):
    RFF_DIM: int = 200
    SIGMA: float = 1.0


class UCBExperimentSetting(ExperimentSetting):
    BETA: float = 1.0
