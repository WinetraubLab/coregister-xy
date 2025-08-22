import numpy as np
import numpy.testing as npt
import unittest

from PCR99a import sRt_from_N_points

class TestPCR99a(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        n_pts = 20
        self.P = np.random.rand(3,n_pts)
        self.s = 1.2

        alpha = np.pi/18 # 10 degrees
        self.R = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        # self.t = np.array([[1], [-2], [3]])
        self.t = np.array([1,-2,3])
        self.Q = self.s * (self.R @ self.P) + self.t.reshape((3,1))

    def test_main_function_runs(self):
        # TODO
        pass
    
    def test_sRt_3points(self):
        s,R,t = sRt_from_N_points(self.P[:,3], self.Q[:,3])
        self.assertAlmostEqual(s, self.s)
        npt.assert_allclose(R, self.R, rtol=1e-4, atol=1e-4)
        npt.assert_allclose(t, self.t, rtol=1e-4, atol=1e-4)

    def test_sRt_3points(self):
        s,R,t = sRt_from_N_points(self.P, self.Q)
        self.assertAlmostEqual(s, self.s)
        npt.assert_allclose(R, self.R, rtol=1e-4, atol=1e-4)
        npt.assert_allclose(t, self.t, rtol=1e-4, atol=1e-4)

