import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

def model_equation_rr(emitter_estimate_position, sat_position, sat_velocity):
    # range rate = rho_dot * rho_hat = (sat_velocity - emitter veloicty) * (sat_position - emitter position)/norm(sat_position - emitter position)
    # Since emitter velocity is 0 we have
    # range rate =  sat_velocity * (sat_position - emitter position)/norm(sat_position - emitter position)
    # since emitter position is unknown, using the current estimated position gives our predicted range rates, thus...
    # predicted_range_rate = sat_velocity_i * (sat_position_i - emitter_estimate_position)/norm(sat_position_i - emitter_estimate_position)
    # REF: Orbit Determination at a Single Ground Station Using Range Rate Data, Daniel Coyle" and Henry J. Pernicka
    predicted_range_rate = np.zeros((sat_position.shape[0], 1))
    for i, sat_pos in enumerate(sat_position):
        predicted_range_rate[i] = np.dot(
            sat_velocity[i], sat_pos[:, np.newaxis] - emitter_estimate_position
        ) / np.linalg.norm(sat_pos[:, np.newaxis] - emitter_estimate_position)
    return predicted_range_rate.flatten()

def model_equation_foa(parameter_estimate,aircraft_position, aircraft_velocity):
    frequency = parameter_estimate[3]
    predicted_frequency = frequency * (
        1
        - (
            model_equation_rr(
                parameter_estimate.flatten()[:3], aircraft_position, aircraft_velocity
            )
            / SPEED_OF_LIGHT
        )
    )
    return predicted_frequency.flatten()


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


def jacobian_foa(parameter_estimate,aircraft_position, aircraft_velocity):
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
        model_equation_rr(
            parameter_estimate[:3], aircraft_position, aircraft_velocity
        )
        / SPEED_OF_LIGHT
    )
    H = np.column_stack([partial_x, partial_y, partial_z, partial_foa])
    return H

