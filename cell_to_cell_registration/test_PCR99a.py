import numpy as np
import numpy.testing as npt
import unittest

from PCR99a import sRt_from_N_points, _score_correspondences, core_PCR99a, plane_ransac, compute_affine

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
        P = np.array([
            [ 2.14,  1.71,  1.87,  1.18,  1.44],
            [-1.93, -1.75, -1.66, -1.51, -1.30],
            [ 3.61,  4.15,  3.10,  3.40,  3.27]])

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
        npt.assert_almost_equal(A, P, decimal=3)
        npt.assert_almost_equal(B, Q, decimal=3)

    def test_ransac2(self):
        P = np.array([
            [ 2.14,  1.71,  1.87,  1.18,  1.44],
            [-1.93, -1.75, -1.66, -1.51, -1.30],
            [ 3.61,  4.15,  3.10,  3.40,  3.27]])
        Q = self.s * (self.R @ P) + self.t.reshape((3,1))

        random_P = np.array([
            [215,  -67, 134, 489, -321, 402],
            [-112,  45, 378, -250, 190, -77],
            [59, 280, -341, 102, 410, -88]
        ])
        random_Q = np.array([
            [-310,  122, -150,  502, -280,  330],
            [ 200, -120,  420, -300,  250,  -40],
            [ -40,  350, -280,  140,  450,  -60]
        ])
        P1 = np.hstack([P[:,:5], random_P])
        Q1 = np.hstack([Q[:,:5], random_Q])
        A, B = plane_ransac(P1,Q1,n_iter=200)

        self.assertAlmostEqual(A.shape[1], 5)
        npt.assert_array_less(A, 20)

    def test_affine(self):
        T = compute_affine(self.P, self.Q)
        pts = np.vstack([self.P, np.ones((1, self.P.shape[1]))])
        P_transformed = T @ pts
        npt.assert_allclose(self.Q, P_transformed)
        