import numpy as np
import numpy.testing as npt
import unittest

from PCR99a import sRt_from_N_points, _score_correspondences, _core_PCR99a, plane_ransac, _compute_affine, calculate_affine_alignment

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

        A, B = _core_PCR99a(P, Q, log_ratio_mat, sort_idx, 10, thr1, sigma, thr2)
        npt.assert_almost_equal(A, P, decimal=3)
        npt.assert_almost_equal(B, Q, decimal=3)

    def test_plane_ransac(self):
        np.random.seed(42)
        P = np.array([
            [ 2.14,  1.71,  1.87,  1.18,  1.44, 5.23, -9.52],
            [-1.93, -1.75, -1.66, -1.51, -1.30, 7.34, 2.13],
            [ 3.61,  4.15,  3.10,  3.40,  3.27, -8.99, -3.21],
            ])

        Q = self.s * (self.R @ P) + self.t.reshape((3,1))
        a,b = plane_ransac(P,Q)
        self.assertAlmostEqual(a.shape[1], 7)

    def test_affine(self):
        T = _compute_affine(self.P, self.Q)
        pts = np.vstack([self.P, np.ones((1, self.P.shape[1]))])
        P_transformed = (T @ pts)[:3]
        npt.assert_allclose(self.Q, P_transformed)

    def test_all(self):
        xyz_oct = np.loadtxt('example_data/oct_points.csv', delimiter=',')
        xyz_hist = np.loadtxt('example_data/fl_points.csv', delimiter=',')
        T, (s,R,t) = calculate_affine_alignment(xyz_oct, xyz_hist, plane_inlier_thresh=5, z_dist_thresh=5,
                 penalty_threshold=8, xy_translation_penalty_weight=1)

        # Matlab result from finding absolute best fit inliers (no early stop)
        s_mat = 1.0114
        R_mat = np.array([
            [ 1.0000,   -0.0017,   -0.0088],
            [ 0.0018 ,   1.0000  ,  0.0041],
            [ 0.0088  , -0.0041   , 1.0000]
            ])
        t_mat = np.array([
            -4.2247,
            5.7450,
            -55.73
            ])
        npt.assert_allclose(R, R_mat,atol=0.01)
        npt.assert_allclose(s, s_mat,atol=0.01)
        npt.assert_allclose(t, t_mat,atol=5)
