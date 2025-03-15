import numpy as np
import scipy
from typing import Callable
# we need measurement
# initial parameters
# initial covariance matrix P
# model equation
# Jacobian matrix
class ExtendedKalmanFilter:
    def __init__(self, 
                 initial_parameters: np.ndarray,
                 initial_covariance: np.ndarray, 
                 measurement: np.ndarray,
                 sensor_noise: np.ndarray,
                 confidence: float, 
                 model: Callable, 
                 linearized_model: Callable):
        """Initialization step of EKF algorithm

        Args:
            initial_parameters (np.ndarray): a 1 x n row vector of initial parameters
            initial_covariance (np.ndarray): A nxn covariance matrix of uncertainty (error covariance matrix)
            measurement (np.ndarray): a 1x1 matrix of the current measurement 
            sensor_noise (np.ndarray): a 1 x n row vector of sensor noise (uncertainty variance)
            confidence (float): A scalar (0 < confidence < 1) used to determine confidence intervals of estimated parameter         
            model (Callable): The model equation used for prediction of parameters
            linearized_model (Callable): H - The jacobian of the model equation

        Raises:
            ValueError: h(x) model equation requires a function
            ValueError: H jacobian matrix requires a function
        """        
        
        self.inital_parameters = initial_parameters
        self.initial_covariance = initial_covariance
        self.sensor_noise = sensor_noise
        self.measurement = measurement
        self.confidence = confidence
        if not callable(model):
            raise ValueError('A model equation is required')
        if not callable(linearized_model):
            raise ValueError('A linearized model is required')
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
        """Provided measurement standard deviations (since we consider measurements to be IID and uncorrelated) only a diagonal matrix
        In the future may need to update this equation"""
        return np.eye(len(self.measurement)) * self.sensor_noise**2
    
    def chisq_k(self):
        """Helper function for confidence interval generation"""
        k = scipy.stats.chi2.ppf(self.confidence, len(self.inital_parameters))
        return k

    def solve_ekf(self, *args, **kwargs):
        """Generate EKF estimate based on a prior parameters, current measurement, and predicted parameters
        """
        R = self.measurement_error_covariance()
        H = self.jacobian(self.inital_parameters, *args, **kwargs)
        K = self.initial_covariance @ H.T @ np.linalg.inv((H @ self.initial_covariance @ H.T) + R)
        h = self.model_equation(self.inital_parameters, *args, **kwargs)
        x_update = self.inital_parameters + K @ (self.measurement - h)
        P_update = (np.eye(len(self.inital_parameters)) - K @ H) @ self.initial_covariance
        self.inital_parameters = x_update
        self.initial_covariance = P_update
        k = self.chisq_k()
        ci_lower = x_update - np.sqrt(k * P_update)
        ci_upper = x_update + np.sqrt(k * P_update)
        return {"ci_lower": ci_lower, "ci_upper": ci_upper, "x_estimate": x_update}
        