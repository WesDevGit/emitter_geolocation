import scipy
import numpy as np

def plot_error_ellipse(P, center=(0,0), confidence=0.95, ax=None):
    # P = UDU.T
    eigenvalues, eigenvectors = np.linalg.eig(P)
    semi_major_vector = eigenvectors[:, eigenvalues.argmax()]
    semi_minor_vector = eigenvectors[:, eigenvalues.argmin()]

    chi2_val = scipy.stats.chi2.ppf(confidence, df=semi_major_vector.size)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array([np.cos(theta), np.sin(theta)])
    D = np.diag(
        np.sqrt(
            np.array(
                [
                    eigenvalues[eigenvalues.argmax()],
                    eigenvalues[eigenvalues.argmin()],
                ]
            )
            * chi2_val
        )
    )
    ellipse_points = (
        np.array([semi_major_vector, semi_minor_vector]).T @ D @ ellipse
    )
    ellipse_points[0, :] += center[0]
    ellipse_points[1, :] += center[1]
    semi_major = semi_major_vector * np.sqrt(
        eigenvalues[eigenvalues.argmax()] * chi2_val
    )
    semi_minor = semi_minor_vector * np.sqrt(
        eigenvalues[eigenvalues.argmin()] * chi2_val
    )
    return ellipse_points, semi_major, semi_minor
    # ax.plot(
    #     ellipse_points[0, :],
    #     ellipse_points[1, :],
    #     'b',
    #     label='95% Confidence Error Ellipse',
    # )
    # ax.scatter(
    #     center[0], center[1], c='red', marker='o', label='Estimated Position'
    # )
    # ax.axhline(0, color='black', linewidth=0.5)
    # ax.axvline(0, color='black', linewidth=0.5)
    # ax.set_aspect('equal', 'box')
