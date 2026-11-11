"""Physical features of the simulator inputs."""
import numpy as np

from .model import D


def feature_map(rho):
    """Linear map from the 23 inputs to the 11 standardised loads and 11 stroke severities."""
    M = np.zeros((D, 22))
    M[0, :11] = np.sqrt(rho)
    M[1:12, :11] = np.sqrt(1 - rho) * np.eye(11)
    M[12:23, 11:] = np.eye(11)
    return M
