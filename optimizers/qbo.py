import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import NormalDistribution
from qiskit_ibm_runtime.fake_provider import FakeWashington
from tqdm import tqdm

from solvers import *

from solvers.solver import Solver


device = FakeWashington()
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)


class QBO:
    def __init__(self, num_iter=10, estimate_function="quantum", solver_instance="ucb",
                 starting_points=None, experiment_settings=None):

        if starting_points is None:
            starting_points = []

        self.results = self.reset_results()
        self.solver_instance = None
        self.set_solver(solver_instance)

        self.experiment_settings = experiment_settings
        self.num_iter = num_iter
        self.starting_points = starting_points

        if estimate_function == "identity":
            self.estimate_function = self.estimate_identity
        elif estimate_function == "quantum":
            self.estimate_function = self.estimate_quantum
        elif estimate_function == "normal":
            self.estimate_function = self.estimate_normal
        else:
            raise ValueError("Unknown estimation function")

        self.total_calls = 0
        self.history_quantum = []
        self.solver = None
        # TODO: parametr sigma z papera: coto

    def set_solver(self, solver_instance):
        if type(solver_instance) is str:
            if solver_instance == "ucb":
                self.solver_instance = UcbGPSolver
            elif solver_instance == "rff":
                self.solver_instance = RFFSolver
            elif solver_instance == "tsgp":
                self.solver_instance = TSGPSolver
            elif solver_instance == "weighted_rff":
                self.solver_instance = RffWeightedGPSolver
            elif solver_instance == "weighted_rff_ts":
                self.solver_instance = TSRffWeightedGPSolver
            elif solver_instance == "weighted_rff_pca":
                self.solver_instance = PcaRffWeightedGPSolver
            elif solver_instance == "weighted_rff_ts_pca":
                print("HERE")
                self.solver_instance = PcaTSRffWeightedGPSolver
            else:
                raise ValueError("Unknown solver instance, co to kurwa jest")
        elif issubclass(solver_instance, Solver):
            self.solver_instance = solver_instance
        else:
            raise ValueError("Unknown solver instance")

    @staticmethod
    def reset_results():
        return {'best': {'max_estimated_value': None,
                         'max_predicted_value': None,
                         'max_parameters': None},
                'history': {'estimated_values': [],
                            'predicted_values': [],
                            'parameters': []},
                'calls': []
                }

    def create_init_points(self, function, parameters_bounds: dict[str, list]):
        default_points = self.starting_points.copy()
        number_of_default_points = (self.experiment_settings.INIT_POINTS - len(default_points))
        starting_points = [np.random.uniform(value[0], value[1], size=number_of_default_points) for
                           key, value in
                           parameters_bounds.items()]
        starting_points = np.array(default_points + starting_points).T.tolist()
        y_init = []
        y_init_true = []
        for point in starting_points:
            output = function(*point)
            estimated_output = self.estimate_function(output)
            y_init.append(estimated_output)
            y_init_true.append(output)
            self.results['history']['parameters'].append(np.array(point))
            self.results['history']['predicted_values'].append(output)
        self.results['history']['estimated_values'] = y_init
        best = np.argmax(np.array(y_init))
        best_true = np.argmax(np.array(y_init_true))
        self.results['best']['max_estimated_value'] = y_init[best]
        self.results['best']['max_predicted_value'] = y_init_true[best_true]
        self.results['best']['max_parameters'] = starting_points[best]

    def find_maximum(self, function_to_optimize, parameters_bounds: dict[str, list]):
        self.results = self.reset_results()
        self.create_init_points(function_to_optimize, parameters_bounds)
        self.solver = self.solver_instance(len(parameters_bounds),
                                           (self.results['history']['parameters'],
                                            self.results['history']['estimated_values']),
                                           self.experiment_settings)

        self.solver.fit()
        pbar = tqdm(total=self.num_iter - self.total_calls)
        while self.total_calls < self.num_iter:
            current_calls = self.total_calls
            new_point = self.solver.best_point()
            output = function_to_optimize(*new_point)
            estimated_output = self.estimate_function(output)

            self.results['history']['parameters'].append(new_point)
            self.results['history']['estimated_values'].append(estimated_output)
            self.results['history']['predicted_values'].append(output)

            if output > self.results['best']['max_predicted_value']:
                self.results['best']['max_estimated_value'] = estimated_output
                self.results['best']['max_parameters'] = new_point
                self.results['best']['max_predicted_value'] = output
            self.solver.fit()
            progress_increment = self.total_calls - current_calls
            pbar.update(progress_increment)
        pbar.close()

    def estimate_identity(self, value, variance=0.05 ** 2):
        self.total_calls += 1
        self.results['calls'].append(1)
        return value

    def estimate_normal(self, value, variance=None):
        if variance is None:
            variance = self.experiment_settings.VARIANCE ** 2
        self.total_calls += 1
        self.results['calls'].append(1)
        return np.random.normal(value, variance)

    def estimate_quantum(self, value, variance=None):
        if variance is None:
            variance = self.experiment_settings.VARIANCE ** 2
        stddev = np.sqrt(variance)
        num_uncertainty_qubits = self.experiment_settings.NUMBER_OF_QUBITS
        c_approx = 1  # TODO: co to jest
        slopes = 1
        offsets = 0

        low = value - 3 * stddev
        high = value + 3 * stddev
        uncertainty_model = NormalDistribution(
            num_uncertainty_qubits,
            mu=value,
            sigma=variance,
            bounds=(low, high)
        )

        linear_payoff = LinearAmplitudeFunction(
            num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(low, high),
            image=(low, high),
            rescaling_factor=c_approx,
        )

        num_qubits = linear_payoff.num_qubits
        monte_carlo = QuantumCircuit(num_qubits)
        monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
        monte_carlo.append(linear_payoff, range(num_qubits))

        try:
            epsilon = self.solver.weights[-1] / (3 * stddev)
        except:
            epsilon = 0.1 / (3 * stddev)

        objective_qubits = [0]
        seed = 0  # TODO: seed

        epsilon = np.clip(epsilon, 1e-6, 0.5)

        alpha = 0.05
        max_shots = 32 * np.log(2 / alpha * np.log2(np.pi / (4 * epsilon)))

        # construct estimation problem. post_processing is the inverse of the rescaling, i.e., it maps the [0,
        # 1] interval to the original one. objective_qubits is the list of qubits that are used to encode the
        # objective function. problem is the estimation problem that is passed to the QAE algorithm.
        problem = EstimationProblem(state_preparation=monte_carlo, objective_qubits=objective_qubits,
                                    post_processing=linear_payoff.post_processing)

        if self.experiment_settings.QUANTUM_NOISE:
            ae = IterativeAmplitudeEstimation(
                epsilon_target=epsilon, alpha=alpha, sampler=Sampler(backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                }, run_options={"shots": int(np.ceil(max_shots)), "seed_simulator": seed},
                    transpile_options={"seed_transpiler": seed}, )
            )
        else:
            ae = IterativeAmplitudeEstimation(
                epsilon_target=epsilon,
                alpha=alpha, sampler=Sampler(
                    run_options={"shots": int(np.ceil(max_shots)),
                                 "seed_simulator": seed}
                )
            )

        # Running result
        result = ae.estimate(problem)
        estimate_value = result.estimation_processed

        num_oracle_queries = result.num_oracle_queries
        if num_oracle_queries == 0:
            num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

        self.total_calls += num_oracle_queries
        self.history_quantum.append(num_oracle_queries)
        self.results['calls'].append(num_oracle_queries)
        return estimate_value
