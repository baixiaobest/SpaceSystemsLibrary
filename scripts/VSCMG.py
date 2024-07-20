''' Modeling of Variable Speed Control Moment Gyroscope'''
import numpy as np
from kinematics.kinematics import euler_321_to_DCM, body_anuglar_velocity_to_321_euler_rates_matrix
from numpy.linalg import inv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_dxdt(Is, J, u):
    '''
    :param Is: Spacecraft inertia matrix, excluding VSCMG, in spacecraft body frame.
    :param J: Inertia matrix of VSCMG, in VSCMG frame.
    :param u: Control input function u(t).
        Given time t, it outputs control vector [gimbal angular acceleration, reaction wheel angular acceleration]
    :return: dxdt
    '''

    def dxdt(t, x):
        yaw, pitch, roll = x[0:3]
        w_B = x[3:6]
        γ = x[6]
        γ_dt = x[7]
        Ω = x[9]

        γ_ddt, Ω_dt = u(t)

        M = body_anuglar_velocity_to_321_euler_rates_matrix(pitch, roll)
        euler_rate = M @ w_B

        # Gimbal frame to body frame.
        R_B_G = np.array([[np.cos(γ), -np.sin(γ), 0],
                          [np.sin(γ), np.cos(γ), 0],
                          [0, 0, 1]])
        g_s = R_B_G[:, 0]
        g_t = R_B_G[:, 1]
        g_g = R_B_G[:, 2]

        w_G = R_B_G.T @ w_B  # Spacecraft angular velocity in gimbal frame
        w_s = w_G[0]
        w_t = w_G[1]
        w_g = w_G[2]

        #  Total inertia in body frame.
        I = Is + R_B_G @ J @ R_B_G.T
        I_inv = inv(I)
        Js = J[0, 0]
        Jt = J[1, 1]
        Jg = J[2, 2]

        # Equation of motion
        w_dt = I_inv @ (- np.cross(w_B, I @ w_B)
                        - g_s * (Js * (Ω_dt + γ_dt * w_t) - (Jt - Jg) * w_t * γ_dt)
                        - g_t * (Js * (w_s + Ω) * γ_dt - (Jt + Jg) * w_s * γ_dt + Js * Ω * w_g)
                        - g_g * (Jg * γ_ddt - Js * Ω * w_t))

        return np.concatenate((euler_rate, w_dt, [γ_dt, γ_ddt, Ω, Ω_dt]))

    return dxdt


def u_func(t):
    '''
    Given time t, return [gimbal angular acceleration, reaction wheel angular acceleration]
    :param t: time
    :return: [gimbal angular acceleration, reaction wheel angular acceleration]
    '''
    return np.array([0, 0])


if __name__ == '__main__':
    x0 = np.array([0, 0, 0,  # yaw, pitch, roll
                   1, 0.01, 0,  # body angular velocity
                   0, 0,  # gimbal angle, gimbal angular rate,
                   0, 5])  # reaction wheel angle, reaction wheel angular rate

    J = np.diag([1, 0.01, 0.01])  # Gyro/reaction wheel inertia.

    Is = np.diag([3, 2, 5])

    dxdt = get_dxdt(Is, J, u_func)

    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * 10)

    solution = solve_ivp(dxdt, t_span, x0, method='RK45', t_eval=t_eval)

    yaw = solution.y[0]
    pitch = solution.y[1]
    roll = solution.y[2]
    w1 = solution.y[3]
    w2 = solution.y[4]
    w3 = solution.y[5]
    γ = solution.y[6]
    γ_dt = solution.y[7]
    θ = solution.y[8]
    Ω = solution.y[9]

    w_N1 = []
    w_N2 = []
    w_N3 = []
    Htotal1 = []
    Htotal2 = []
    Htotal3 = []

    for i in range(len(solution.t)):
        R_B_N = euler_321_to_DCM(yaw[i], pitch[i], roll[i])
        w_B = np.array([w1[i], w2[i], w3[i]])
        w_N = R_B_N.T @ w_B
        w_N1.append(w_N[0])
        w_N2.append(w_N[1])
        w_N3.append(w_N[2])

        R_B_G = np.array([[np.cos(γ[i]), -np.sin(γ[i]), 0],
                          [np.sin(γ[i]), np.cos(γ[i]), 0],
                          [0, 0, 1]])

        HS_B = Is @ w_B  # Spacecraft angular momentum in body frame, excluding VSCMG
        wG_B = R_B_G @ np.array([Ω[i], 0, γ_dt[i]]) + w_B  # VSCMG angular velocity in B frame.
        J_B = R_B_G @ J @ R_B_G.T  # VSCMG inertia matrix in body frame.
        HG_B = J_B @ wG_B   # VSCMG angular momentum in body frame.

        Htotal_B = HS_B + HG_B
        Htotal_N = R_B_N.T @ Htotal_B

        Htotal1.append(Htotal_N[0])
        Htotal2.append(Htotal_N[1])
        Htotal3.append(Htotal_N[2])

    # Plot the results
    plt.figure(figsize=(6, 4))

    plt.subplot(3, 1, 1)
    plt.plot(solution.t, yaw, label='Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw [rad]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(solution.t, pitch, label='Pitch')
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch [rad]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(solution.t, roll, label='Roll')
    plt.xlabel('Time [s]')
    plt.ylabel('Roll [rad]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Anuglar velocity
    plt.figure(figsize=(6, 4))

    plt.subplot(3, 1, 1)
    plt.plot(solution.t, w1, label='W_B1')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B1 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(solution.t, w2, label='W_B2')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B2 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(solution.t, w3, label='W_B3')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B3 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # VSCMG
    plt.figure(figsize=(6, 4))

    plt.subplot(2, 2, 1)
    plt.plot(solution.t, [u_func(t)[0] for t in solution.t], label='gimbal angular acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('γ_ddt [rad/s/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(solution.t, γ_dt, label='gimbal angular velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('γ_dt [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(solution.t, [u_func(t)[1] for t in solution.t], label='reaction wheel angular acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Ω_dt [rad/s/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(solution.t, Ω, label='reaction wheel angular velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Ω [rad/s]')
    plt.grid(True)
    plt.legend()

    #  Angular momentum
    plt.figure(figsize=(6, 4))

    plt.subplot(3, 1, 1)
    plt.plot(solution.t, Htotal1, label='Spacecraft angular momentum H1')
    plt.xlabel('Time [s]')
    plt.ylabel('H1 [kg*m^2/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(solution.t, Htotal2, label='Spacecraft angular momentum H2')
    plt.xlabel('Time [s]')
    plt.ylabel('H2 [kg*m^2/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(solution.t, Htotal3, label='Spacecraft angular momentum H3')
    plt.xlabel('Time [s]')
    plt.ylabel('H3 [kg*m^2/s]')
    plt.grid(True)
    plt.legend()

    # Ploting animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    def get_cube():
        # Define the vertices of a cube
        r = [-0.5, 0.5]
        vertices = np.array([[x, y, z] for x in r for y in r for z in r])
        return vertices


    def get_faces(vertices):
        # Define the 6 faces of the cube
        faces = [[vertices[j] for j in [0, 1, 3, 2]],
                 [vertices[j] for j in [4, 5, 7, 6]],
                 [vertices[j] for j in [0, 1, 5, 4]],
                 [vertices[j] for j in [2, 3, 7, 6]],
                 [vertices[j] for j in [0, 2, 6, 4]],
                 [vertices[j] for j in [1, 3, 7, 5]]]
        return faces


    vertices = get_cube()
    face_colors = ['cyan', 'magenta', 'yellow', 'blue', 'green', 'red']


    def animate(i):
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Apply rotation
        R_B_N = euler_321_to_DCM(yaw[i], pitch[i], roll[i])
        rotated_vertices = [R_B_N.T @ v for v in vertices]
        faces = get_faces(rotated_vertices)
        poly3d = [list(map(tuple, face)) for face in faces]

        for poly, color in zip(poly3d, face_colors):
            collection = Poly3DCollection([poly], facecolors=color, linewidths=1, edgecolors='r', alpha=.75)
            ax.add_collection3d(collection)

        # Plot angular momentum vector
        ax.quiver(0, 0, 0, Htotal1[i], Htotal2[i], Htotal3[i], color='black', label='Angular Momentum')

        ax.quiver(0, 0, 0, w_N1[i], w_N2[i], w_N3[i], color='red', label="Angular Velocity")

        b1 = R_B_N.T @ np.array([0, 0, 1])
        ax.quiver(0, 0, 0, b1[0], b1[1], b1[2], color='green', label="Body frame b3")

        ax.set_title(f'Time = {solution.t[i]:.2f}s')
        ax.legend()
        return fig,


    ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=50, blit=False)
    plt.show()
