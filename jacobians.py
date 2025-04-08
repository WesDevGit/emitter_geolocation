import numpy as np
import model_equations

SPEED_OF_LIGHT = 299_792_458.0


def range_rate_jacobian(emitter_estimate_position, sat_position, sat_velocity):
    # since our emitter is stationary we can omit the partial wrt to velocity vector columns so H is Nx3 not Nx6
    norm = np.linalg.norm(sat_position - emitter_estimate_position.T, axis=1)
    r_R = (
        (sat_velocity[:, 0] * (sat_position[:, 0] - emitter_estimate_position[0]))
        + (sat_velocity[:, 1] * (sat_position[:, 1] - emitter_estimate_position[1]))
        + (sat_velocity[:, 2] * (sat_position[:, 2] - emitter_estimate_position[2]))
    )
    partial_x = -sat_velocity[:, 0] / norm
    -((sat_position[:, 0] - emitter_estimate_position[0]) / (norm**3)) * r_R
    partial_y = -sat_velocity[:, 1] / norm
    -((sat_position[:, 1] - emitter_estimate_position[1]) / (norm**3)) * r_R
    partial_z = -sat_velocity[:, 2] / norm
    -((sat_position[:, 2] - emitter_estimate_position[2]) / (norm**3)) * r_R
    H = np.column_stack([partial_x, partial_y, partial_z])
    return H


def jacobian_foa(parameter_estimate, aircraft_position, aircraft_velocity):
    """"""
    parameter_estimate = parameter_estimate.flatten()
    norm = np.linalg.norm(aircraft_position - parameter_estimate[:3], axis=1)
    r_R = (
        (aircraft_velocity[:, 0] * (aircraft_position[:, 0] - parameter_estimate[0]))
        + (aircraft_velocity[:, 1] * (aircraft_position[:, 1] - parameter_estimate[1]))
        + (aircraft_velocity[:, 2] * (aircraft_position[:, 2] - parameter_estimate[2]))
    )

    partial_x = (
        parameter_estimate[3]
        * (-1 / SPEED_OF_LIGHT)
        * (
            -aircraft_velocity[:, 0] / norm
            + ((aircraft_position[:, 0] - parameter_estimate[0]) / (norm**3)) * r_R
        )
    )
    partial_y = (
        parameter_estimate[3]
        * (-1 / SPEED_OF_LIGHT)
        * (
            -aircraft_velocity[:, 1] / norm
            + ((aircraft_position[:, 1] - parameter_estimate[1]) / (norm**3)) * r_R
        )
    )
    partial_z = (
        parameter_estimate[3]
        * (-1 / SPEED_OF_LIGHT)
        * (
            -aircraft_velocity[:, 2] / norm
            + ((aircraft_position[:, 2] - parameter_estimate[2]) / (norm**3)) * r_R
        )
    )
    partial_foa = 1 - (
        model_equations.model_equation_rr(
            parameter_estimate[:3], aircraft_position, aircraft_velocity
        )
        / SPEED_OF_LIGHT
    )
    H = np.column_stack([partial_x, partial_y, partial_z, partial_foa])
    return H


def jacobian_doa(estimate_location, sensor_location):
    rng = (estimate_location[0] - sensor_location[:,0])**2 + (estimate_location[1] - sensor_location[:,1])**2
    partial_x = -(1/rng)*(estimate_location[1] - sensor_location[:,1])
    partial_y = (1/rng)*(estimate_location[0] - sensor_location[:,0])
    partial_b = np.repeat(1, len(sensor_location))
    return np.column_stack([partial_x, partial_y, partial_b])