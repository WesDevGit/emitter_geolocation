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


def model_equation_range_difference(x_old, sensor_positions):
    """TDOA Model"""
    psedorange_estimate = np.array(
        [
            np.linalg.norm(x_old - sensor_positions[1])
            - np.linalg.norm(x_old - sensor_positions[0]),  # emitter 2,1
            np.linalg.norm(x_old - sensor_positions[2])
            - np.linalg.norm(x_old - sensor_positions[0]),  # emitter 3,1
            np.linalg.norm(x_old - sensor_positions[3])
            - np.linalg.norm(x_old - sensor_positions[0]),  # emitter 4,1
        ]
    )

    return psedorange_estimate

def model_equation_doa(x_old, aircraft_positions):
    """DOA Model"""
    angles = np.arctan2(
        x_old[1] - aircraft_positions[:, 1],
        x_old[0] - aircraft_positions[:, 0],
    )
    return angles

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



