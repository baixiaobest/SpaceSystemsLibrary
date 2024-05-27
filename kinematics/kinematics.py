import numpy as np
from kinematics.KinematicStruct import KinematicTransform, KinematicChain

def velocity_transport_theorem(p: np.ndarray, v: np.ndarray, omega: np.ndarray, v_BN: np.ndarray) -> np.ndarray:
    '''
    Transform the particle velocity vector v relative to frame B to the one relative to frame N.
    B and N are rotating and translating relative to each other.

    Note: v, omega and v_BN vectors need to be expressed in the same reference frame,
    in frame B or N or any other frame.

    :param p: The vector of a particle, from origin of frame B to the particle.
    :param v: The velocity of the particle relative to frame B.
    :param omega: The rotational vector of B relative to N.
    :param v_BN: The velocity of B relative to N.
    :return: The time derivative of a vector relative to frame N. It will be expressed
    in the frame in which the input vectors are expressed in.
    '''

    return v + v_BN + np.cross(omega, p)

def acceleration_transport_theorem(
        p: np.ndarray, v: np.ndarray, a: np.ndarray, omega: np.ndarray,
        alpha: np.ndarray, a_BN: np.ndarray) -> np.ndarray:
    '''
    Transform the acceleration vector relative to frame B to relative to frame N.
    B and N are rotating and translating relative to each other.

    Note: All the vectors need to be expressed in the same reference frame.

    :param p: The vector of a particle, from origin of frame B to the particle.
    :param v: The velocity of the particle relative to frame B.
    :param a: The acceleration of the particle relative to frame B.
    :param omega: The rotational vector of B relative to N.
    :param alpha: The time derivative of omega.
    :param a_BN: The acceleration of B relative to N.
    :return: The acceleration of vector relative to frame N. It will be expressed
    in the frame in which the input vectors are expressed in.
    '''

    return a_BN + a + np.cross(alpha, p) + 2 * np.cross(omega, v) + np.cross(omega, np.cross(omega, p))

def position_kinematic_chain(kinematic_chain: KinematicChain):
    '''
    Compute the position vector from the origin of the first coordinate system to
    the origin of the last coordinate system.
    :param kinematic_chain: A list of kinematic transformations.
    :return: The position vector from the origin of the first coordinate system to
    the origin of the last coordinate system.
    '''

    n = len(kinematic_chain)
    p = np.zeros(3)
    for i in reversed(range(0, n)):
        transform = kinematic_chain[i]
        p = transform.R @ p
        p = p + transform.t
    return p


def velocity_transport_kinematic_chain(kinematic_chain: KinematicChain):
    '''
    Get the velocity of last coordinate system relative to the first one.
    :param kinematic_chain: A list of kinematic transformations.
    :return: The velocity of the last coordinate system relative to the first one.
    '''
    n = len(kinematic_chain)
    p = np.zeros(3)
    v = np.zeros(3)
    for i in reversed(range(0, n)):
        transform = kinematic_chain[i]
        v = transform.R @ v
        p = transform.R @ p
        v = velocity_transport_theorem(p, v, transform.omega, transform.v)
        p += transform.t

    return v

def acceleration_transport_kinematic_chain(kinematic_chain: KinematicChain):
    '''
    Get the acceleration of last coordinate system relative to the first one.
    :param kinematic_chain: A list of kinematic transformations.
    :return: The acceleration of the last coordinate system relative to the first one.
    '''

    n = len(kinematic_chain)
    p = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    for i in reversed(range(0, n)):
        transform = kinematic_chain[i]
        v = transform.R @ v
        p = transform.R @ p
        a = transform.R @ a
        a = acceleration_transport_theorem(p, v, a, transform.omega, transform.alpha, transform.a)
        v = velocity_transport_theorem(p, v, transform.omega, transform.v)
        p += transform.t

    return a