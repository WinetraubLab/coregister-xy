import numpy as np
import numpy.testing as npt
import unittest

from PCR99a import sRt_from_N_points, _score_correspondences, core_PCR99a

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
    
    def test_sRt_3points(self):
        s,R,t = sRt_from_N_points(self.P[:,:3], self.Q[:,:3])
        self.assertAlmostEqual(s, self.s)
        npt.assert_allclose(R, self.R, rtol=1e-4, atol=1e-4)
        npt.assert_allclose(t, self.t, rtol=1e-4, atol=1e-4)

    def test_sRt_Npoints(self):
        s,R,t = sRt_from_N_points(self.P, self.Q)
        self.assertAlmostEqual(s, self.s)
        npt.assert_allclose(R, self.R, rtol=1e-4, atol=1e-4)
        npt.assert_allclose(t, self.t, rtol=1e-4, atol=1e-4)

    def test_step2(self):
        # Known test result from Matlab version
        vals = np.array([
            2.33721474e-08, 5.31067768e-09, 3.21212928e-08, 9.81710382e-09
        ])
        log_ratios = np.array([
            [np.nan,       -0.18232156, -0.18232158, -0.18232156],
            [-0.18232156,  np.nan,      -0.18232155, -0.18232155],
            [-0.18232158, -0.18232155,  np.nan,      -0.18232156],
            [-0.18232156, -0.18232155, -0.18232156,  np.nan]
        ])

        min_costs = _score_correspondences(log_ratios, 0.03)
        npt.assert_allclose(min_costs, vals, rtol=1e-7, atol=1e-7)

    def test_pcr_core(self):
        P = np.array([[ 2.14085717,  1.71839018,  1.87839273,  1.18722237,  1.44944814],
        [-1.93833432, -1.75652702, -1.66191823, -1.5149545 , -1.30236109],
        [ 3.6142523 ,  4.15094862,  3.10151576,  3.4008531 ,  3.27171761]])

        Q = self.s * (self.R @ P) + self.t.reshape((3,1))

        thr1 = 0.03
        thr2 = 5
        sigma = 2

        d_gt = np.sum((Q[:, :, None] - Q[:, None, :])**2, axis=0)  
        d_est = np.sum((P[:, :, None] - P[:, None, :])**2, axis=0) 
        log_ratio_mat = 0.5 * np.log(d_est / d_gt)

        min_costs = _score_correspondences(log_ratio_mat, thr1)
        sort_idx = np.argsort(min_costs)

        A, B = core_PCR99a(P, Q, log_ratio_mat, sort_idx, 10, thr1, sigma, thr2)
        gt_A = np.array([
            [ 2.14085717,  1.71839018,  1.87839273,  1.18722237,  1.44944814],
            [-1.93833432, -1.75652702, -1.66191823, -1.5149545,  -1.30236109],
            [ 3.6142523,   4.15094862,  3.10151576,  3.4008531 ,  3.27171761]])
        gt_B = self.s * (self.R @ gt_A) + self.t.reshape((3,1))
        npt.assert_almost_equal(A, gt_A, decimal=3)
        npt.assert_almost_equal(B, gt_B, decimal=3)
