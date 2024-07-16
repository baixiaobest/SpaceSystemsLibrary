import numpy as np
from scipy.integrate import solve_ivp
from kinematics.kinematics import euler_321_to_DCM
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_dxdt(I, L):
    '''
    :param I: Inertial matrix.
    :param L: Torque function, L(t) gives the torque on rigid body as function of time. L is in body frame.
    :return: dxdt
    '''
    I_inv = inv(I)

    def dxdt(t, x):
        '''
        This method is not very numerically stable. See free_body_rotation2 for more stable method.
        :param t: time.
        :param x: [yaw, pitch, roll, angular momentum on x y z in inertial frame]
        :return: dxdt
        '''
        yaw, pitch, roll = x[0:3]
        H_N = x[3:6]
        R_B_N = euler_321_to_DCM(x[0], x[1], x[2])
        H_B = R_B_N @ H_N
        w_B = I_inv @ H_B

        s_pitch = np.sin(pitch)
        c_pitch = np.cos(pitch)
        s_roll = np.sin(roll)
        c_roll = np.cos(roll)
        M = 1.0/c_pitch * np.array([
            [0, s_roll,               c_roll],
            [0, c_roll*c_pitch, -s_roll*c_pitch],
            [c_pitch,           s_roll*s_pitch, c_roll*s_pitch]])

        return np.concatenate((M @ w_B, R_B_N.T@L(t)))

    return dxdt

def L(t):
    if t < -5:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 0, 0])

if __name__=="__main__":
    I = np.diag([2, 2, 1])
    I_inv = inv(I)

    w0 = np.array([-0.1, -0.1, 1]) # initial body frame angular velocity

    yaw0 = 0
    pitch0 = 0
    roll0 = 0

    H0_B = I @ w0  # initial body frame angular momentum
    R_B_N = euler_321_to_DCM(yaw0, pitch0, roll0)
    H0_N = R_B_N.T @ H0_B  # initial inertial frame angular momentum

    x0 = np.array([yaw0, pitch0, roll0, H0_N[0], H0_N[1], H0_N[2]]) # yaw, pitch, roll, angular momentum H1, H2, H3 in inertial frame.

    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) * 10)

    dxdt_func = get_dxdt(I, L)

    solution = solve_ivp(dxdt_func, t_span, x0, method='RK45', t_eval=t_eval)

    # Extract the solution for roll, pitch, and yaw
    yaw = solution.y[0]
    pitch = solution.y[1]
    roll = solution.y[2]
    H1 = solution.y[3]
    H2 = solution.y[4]
    H3 = solution.y[5]
    w_N1 = []
    w_N2 = []
    w_N3 = []
    w_B1 = []
    w_B2 = []
    w_B3 = []

    for i in range(len(solution.t)):
        R_B_N = euler_321_to_DCM(yaw[i], pitch[i], roll[i])
        H_N = np.array([H1[i], H2[i], H3[i]])
        H_B = R_B_N @ H_N
        w_B = I_inv @ H_B
        w_N = R_B_N.T @ w_B
        w_N1.append(w_N[0])
        w_N2.append(w_N[1])
        w_N3.append(w_N[2])
        w_B1.append(w_B[0])
        w_B2.append(w_B[1])
        w_B3.append(w_B[2])

    # Plot the results
    plt.figure(figsize=(12, 8))

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
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(solution.t, w_B1, label='W_B1')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B1 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(solution.t, w_B2, label='W_B2')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B2 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(solution.t, w_B3, label='W_B3')
    plt.xlabel('Time [s]')
    plt.ylabel('W_B3 [rad/s]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

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
        ax.quiver(0, 0, 0, H1[i], H2[i], H3[i], color='black', label='Angular Momentum')

        ax.quiver(0, 0, 0, w_N1[i], w_N2[i], w_N3[i], color='red', label="Angular Velocity")

        b1 = R_B_N.T @ np.array([0, 0, 1])
        ax.quiver(0, 0, 0, b1[0], b1[1], b1[2], color='green', label="Body frame b3")

        ax.set_title(f'Time = {solution.t[i]:.2f}s')
        ax.legend()
        return fig,


    ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=50, blit=False)
    plt.show()