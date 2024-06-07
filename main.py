from kinematics.kinematics import *
from kinematics.KinematicStruct import *
import sympy as sp

def example_1_4():
    n1, n2, n3, e1, e2, e3, e, n, u, γ, γ_dt, φ, φ_dt, φ_ddt, Re = \
        sp.symbols('n1 n2 n3 e1 e2 e3 e n u γ γ_dt φ φ_dt φ_ddt Re')

    zero = sp.Matrix([0, 0, 0])

    R_γ = sp.Matrix([[sp.cos(γ), -sp.sin(γ), 0],
                     [sp.sin(γ), sp.cos(γ), 0],
                     [0, 0, 0]])

    omega1 = sp.Matrix([0, 0, γ_dt])
    alpha1 = sp.Matrix([0, 0, 0])

    T1 = KinematicTransform(zero, R_γ, omega1, alpha1, zero, zero)

    R_φ = sp.Matrix([[sp.cos(φ), 0, sp.sin(φ)],
                     [0, 1, 0],
                     [-sp.sin(φ), 0, sp.cos(φ)]])

    t = sp.Matrix([Re*sp.cos(φ), 0, Re*sp.sin(φ)])

    T2 = KinematicTransform(t, R_φ, zero, zero, zero, zero)

    omega3 = sp.Matrix([0, -φ_dt, 0])
    alpha3 = sp.Matrix([0, -φ_ddt, 0])
    v3 = sp.Matrix([0, φ_dt * Re, 0])
    a3 = sp.Matrix([0, φ_ddt * Re, 0])

    T3 = KinematicTransform(zero, sp.eye(3), omega3, alpha3, v3, a3)

    KC = KinematicChain([T1, T2, T3])

    p = position_kinematic_chain(KC)
    v = velocity_transport_kinematic_chain(KC)
    a = acceleration_transport_kinematic_chain(KC)
    print(p)
    print(v)
    print(a)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    example_1_4()
