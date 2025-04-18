import numpy as np
import scipy
from typing import Callable


class IteratedLeastSquares:
    """Solve nonlinear least squares problem

    Example Input and output
    obj = IteratedLeastSquares(initial_parameters=np.array([x_naught])[:, np.newaxis],
                            measurements=z,
                            sensor_noise=np.array([sigma]),
                            tol = 1e-3,
                            max_iterations=50,
                            model=h_model [Callable],
                            linearized_model=jac_matrix [Callable])

    x_estimate, P = obj.solve_ils(t, u , y_naught)

    print(np.round(x_estimate, 5).item(), np.round(np.sqrt(P),5).item())
    Converged to a solution in 4 iterations
    0.16815 0.00296
    """

    def __init__(
        self,
        initial_parameters: np.ndarray,
        measurements: np.ndarray,
        measurement_noise: np.ndarray,
        tol: float,
        max_iterations: int,
        model: Callable,
        linearized_model: Callable,
    ):

        self.initial_parameters = initial_parameters
        self.measurement_noise = measurement_noise
        self.tol = tol
        self.max_iterations = max_iterations
        self.measurements = measurements
        if not callable(model):
            raise ValueError("A model equation is required")
        if not callable(linearized_model):
            raise ValueError("A linearized model is required")

        self.model = model
        self.linearized_model = linearized_model

    def model_equation(self, *args, **kwargs):
        """Provided Model equation h(x_current) [Mathematical model of the measurement]

        Given an initial or currrent parameter vector return the predicted parameters (x^hat)
        """

        return self.model(*args, **kwargs)

    def jacobian(self, *args, **kwargs):
        """Provided Linear approximation using the jacobian of model equation

        Returns: A jacobian matrix H
        """
        return self.linearized_model(*args, **kwargs)

    def measurement_error_covariance(self):
        n = self.measurements.size  # or len(self.measurements)
        return np.eye(n) * (self.measurement_noise**2)

    def covariance_matrix_P(self, H: np.ndarray):
        """Estimation error covariance matrix"""
        R = (
            self.measurement_error_covariance()
        )  # measurement error covariance matrix (positive definite)
        R_inv = np.linalg.pinv(R)
        P = np.linalg.pinv(H.T @ R_inv @ H)
        return P

    def iteration(self, x_current, measurements, *args, **kwargs):
        predicted = self.model_equation(x_current, *args, **kwargs)
        residuals = measurements - predicted[:, np.newaxis]
        H = self.jacobian(x_current, *args, **kwargs)
        P = self.covariance_matrix_P(H)
        R = self.measurement_error_covariance()  # sensor noise
        R_inv = np.linalg.pinv(R)
        K = P @ H.T @ R_inv
        correction_term = K @ residuals  # gauss-markov solution
        x_hat = (
            x_current + correction_term
        )  # current parameter estimate + correction term = x_estimate
        return x_hat, P

    def solve_ils(self, *args, **kwargs):
        """Returns estimated parameter and covariance"""
        x_current = self.initial_parameters.copy()
        for iter in range(self.max_iterations):
            x_estimate, P = self.iteration(
                x_current, self.measurements, *args, **kwargs
            )
            if np.linalg.norm(x_estimate - x_current) < self.tol:
                print(f"Converged to a solution in {iter + 1} iterations")

                return x_estimate, P

            if iter == self.max_iterations - 1:
                return print(f"Failed to converage after {iter + 1} iterations.")
            x_current = x_estimate

        return x_current


def model_equation_range_rate(grid_point, aircraft_positions, aircraft_velocity):
    grid_emitter = np.array([grid_point[0], grid_point[1], 0.0])
    range_rate = np.zeros((aircraft_positions.shape[0], 1))
    for i, aircraft_position in enumerate(aircraft_positions):
        range_rate[i] = np.dot(
            aircraft_velocity[i], aircraft_position - grid_emitter
        ) / np.linalg.norm(aircraft_position - grid_emitter)
    return range_rate
