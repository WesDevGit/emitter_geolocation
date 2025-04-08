import numpy as np


def tdoa(tau, p1, p2, partb):
    """TDOA hyperboldas in 2D"""
    c = 299_792.458
    theta = np.arctan2((p2[1] - p1[1]), p2[0] - p1[0])
    tau = tau * 1 / 1000000
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = 0.5 * (p2 + p1)
    c_prime = 0.5 * np.linalg.norm(p2 - p1)
    tau_c = tau * c
    a = np.abs(tau_c) / 2
    b = np.sqrt(c_prime**2 - a**2)
    x_i = np.linspace(-4, 4, 400000)
    if not partb:
        if tau > 0:
            valid_mask = x_i >= a
        else:
            valid_mask = x_i <= -a
    else:
        valid_mask = (x_i <= -a) | (x_i >= a)

    x_valid = x_i[valid_mask]
    inside = (x_valid**2 / a**2) - 1
    y_pos = b * np.sqrt(inside)
    y_neg = -b * np.sqrt(inside)

    h_vector_positive = np.vstack((x_valid, y_pos))
    h_vector_negative = np.vstack((x_valid, y_neg))
    hyperbola_vector_positive = r @ h_vector_positive + m.reshape(2, 1)
    hyperbola_vector_negative = r @ h_vector_negative + m.reshape(2, 1)

    return hyperbola_vector_positive, hyperbola_vector_negative
