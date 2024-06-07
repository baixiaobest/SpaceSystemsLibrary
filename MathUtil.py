import numpy as np
import sympy as sp
from kinematics.KinematicStruct import KinematicChain

def cross_product(a, b):
    """
    Compute the cross product of two vectors, supporting both NumPy arrays and SymPy matrices.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.cross(a, b)
    elif isinstance(a, (sp.Matrix, sp.ImmutableMatrix)) and isinstance(b, (sp.Matrix, sp.ImmutableMatrix)):
        return a.cross(b)
    else:
        raise TypeError("Input vectors must be both numpy arrays or both sympy matrices.")


def is_sympy(kinematic_chain: KinematicChain):
    if len(kinematic_chain) > 0:
        for T in kinematic_chain:
            if not (isinstance(T.t, (sp.Matrix, sp.ImmutableMatrix)) and
                    isinstance(T.R, (sp.Matrix, sp.ImmutableMatrix)) and
                    isinstance(T.omega, (sp.Matrix, sp.ImmutableMatrix)) and
                    isinstance(T.alpha, (sp.Matrix, sp.ImmutableMatrix)) and
                    isinstance(T.v, (sp.Matrix, sp.ImmutableMatrix)) and
                    isinstance(T.a, (sp.Matrix, sp.ImmutableMatrix))):
                return False
        return True
    return False
