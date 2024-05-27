import unittest
from kinematics.kinematics import *
from kinematics.KinematicStruct import *
import numpy as np

class MyTestCase(unittest.TestCase):
    def setUp(self):
        '''
        We have two frames A & B. B relative to A vector p_BA = [10, 0, 0], expressed in A.
        B is rotated counter-clockwise 30 deg relative to A.
        B is rotating relative to A at omega (expressed in A) and accelerating at alpha.
        B is translating v_BA = [2, -2, 0] (expressed in A) and accelerating a_BA relative to A.
        A particle P has relative position p=[3, 5, 0] (expressed in B) to the frame B.
        This particle is moving relative to frame B at v_p = [1, 1, 0] (expressed in B) and
        accelerating at a_BA (expressed in B).
        We are calculating the relative velocity and acceleration of this particle to frame A.
        '''
        self.p = np.array([3, 5, 0])  # in frame B
        self.v_p = np.array([1, 1, 0])  # in frame B
        self.a_p = np.array([-1, 0, 0]) # in frame B
        self.omega = np.array([0, 0, 5])  # in frame A
        self.alpha = np.array([0, 0, 1]) # in frame A
        self.p_BA = np.array([10, 0, 0])  # in frame A
        self.v_BA = np.array([2, -2, 0])  # in frame A
        self.a_BA = np.array([1, 2, 0]) # in frame A
        self.s = np.sin(np.pi / 6)
        self.c = np.cos(np.pi / 6)
        self.T = np.array([[self.c, -self.s, 0],
                          [self.s, self.c, 0],
                          [0, 0, 1]])

        T1 = KinematicTransform(self.p_BA, self.T, self.omega, self.alpha, self.v_BA, self.a_BA)
        T2 = KinematicTransform(self.p, np.identity(3), np.zeros(3), np.zeros(3), self.v_p, self.a_p)
        self.KC = KinematicChain()
        self.KC.add_transform(T1)
        self.KC.add_transform(T2)

    def test_velocity_transport_theorem(self):
        v_p_A = velocity_transport_theorem(self.T@self.p, self.T@self.v_p, self.omega, self.v_BA)

        v_p_A_result = np.array([2 - 16*self.s - 24*self.c, -2 - 24*self.s + 16*self.c, 0])

        np.testing.assert_almost_equal(v_p_A, v_p_A_result, decimal=5)

    def test_acceleration_transport_theorem(self):
        a_p_A = acceleration_transport_theorem(self.T@self.p, self.T@self.v_p, self.T@self.a_p,
                                               self.omega, self.alpha, self.a_BA)
        a_p_A_result = np.array([1 - 91*self.c + 112*self.s, 2 - 112*self.c - 91*self.s, 0])

        np.testing.assert_almost_equal(a_p_A, a_p_A_result, decimal=5)

    def test_position_kinematic_chain(self):
        p = position_kinematic_chain(self.KC)
        p_result = np.array([10 + 3*self.c - 5*self.s, 3*self.s + 5*self.c, 0])
        np.testing.assert_almost_equal(p, p_result, decimal=5)

    def test_velocity_kinematic_chain(self):
        v = velocity_transport_kinematic_chain(self.KC)
        v_result = np.array([2 - 16*self.s - 24*self.c, -2 - 24*self.s + 16*self.c, 0])
        np.testing.assert_almost_equal(v, v_result, decimal=5)

    def test_acceleration_kinematic_chain(self):
        a = acceleration_transport_kinematic_chain(self.KC)
        a_result = np.array([1 - 91*self.c + 112*self.s, 2 - 112*self.c - 91*self.s, 0])
        np.testing.assert_almost_equal(a, a_result, decimal=5)

if __name__ == '__main__':
    unittest.main()
