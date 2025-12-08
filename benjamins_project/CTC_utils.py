from math import pi, acos, cos, sin
import numpy as np
from numpy.typing import NDArray


def randomrotationmatrix(seed: float) -> np.ndarray:
    """Generate random 3D rotation matrix according to ZYX Euler angles

    Args:
        float seed: used to seed the function for reproducability

    Returns:
        3x3 np.array of Euler angles
    """
    psi = seed * 2 * pi
    theta = 0
    phi = acos(1 - 2 * seed)

    Rz = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    Ry = np.array(
        [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    )
    Rx = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    return Rz @ Ry @ Rx


def lennartjones_potential(dist: float, sigma_LJ: float, kB: float) -> float:
    """Calculate Lennard-Jones potential energy
    Args:
        float distance: distance between two atoms [m]
    Returns:
        float: Lennard-Jones potential energy [J]
    """
    K_to_EV = 8.617333262145e-5  # eV/K
    ev_to_K = 1.60217667e-19  # K/eV

    epsilon = 34 * K_to_EV * ev_to_K  # Depth of the potential well [J]

    potential = 4 * epsilon * ((sigma_LJ / dist) ** 12 - (sigma_LJ / dist) ** 6)

    return potential


def lennartjones_force(dist: float, sigma_LJ: float, kB: float) -> float:
    """Calculate the 12-6 Lennard-Jones force
    Args:
        float distance: distance between two atoms [m]
    Returns:
        float: Lennard-Jones force [N]
    """
    K_to_EV = 8.617333262145e-5  # eV/K
    ev_to_K = 1.60217667e-19  # K/eV
    epsilon = 34 * K_to_EV * ev_to_K  # Depth of the potential well [J]
    force = (
        -4
        * epsilon
        * ((6 * sigma_LJ**6) / (dist**7) - (12 * sigma_LJ**12) / (dist**13))
    )
    return force


def intraatomic_force(
    xi: NDArray, xj: NDArray, sigma_LJ: float, kB: float
) -> np.ndarray:
    """
    Computes the intra-atomic force between two atoms.

    Inputs:
      xi, xj : (3,) numpy arrays representing position vectors (X, Y, Z)

    Outputs:
      fij    : (3,) numpy array containing the interatomic forces
    """
    # Calculate interatomic distance
    drij = np.linalg.norm(xi - xj)

    # Geometrical computations
    # xi[0:2] takes the first two elements (X, Y)
    drijxy = np.linalg.norm(xi[0:2] - xj[0:2])

    # Indices: 0->X, 1->Y, 2->Z
    theta_ij = np.arctan2(xj[2] - xi[2], drijxy)
    phi_ij = np.arctan2(xj[1] - xi[1], xj[0] - xi[0])

    # Compute net force magnitude (Assumes LJ function is defined)
    f_mag = lennartjones_force(float(drij), sigma_LJ, kB)

    # Decompose the force
    f_z = np.sin(theta_ij) * f_mag
    f_xy = np.cos(theta_ij) * f_mag

    f_x = np.cos(phi_ij) * f_xy
    f_y = np.sin(phi_ij) * f_xy

    return np.array([-f_x, -f_y, -f_z])
