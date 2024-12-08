
import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane

class TestFitPlane(unittest.TestCase):

    def setUp(self):
        self.template_center_positions_uv_pix = [[0,1],[1,0],[1,1]]

        # rotation around y axis.
        self.template_center_positions_xyz_um_1 = [[0,1,0], [0.9994, 0, -0.0349], [0.9994,1,-0.0349]] 

        # rotation around z axis. 
        self.template_center_positions_xyz_um_2 = [[0.035, 0.9994,5], [0.9994, -0.0349, 5], [1.0343, 0.9645, 5]] 

        # rotation around x axis.
        self.template_center_positions_xyz_um_3 = [[0,0.9994,0.0349], [0.9994,0.0012,-0.0349], [0.9994,1.0006,0.00002]]

    def test_main_function_runs(self):
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um_2, print_inputs=False)
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um_2, print_inputs=True)

    def test_fit_mapping(self):
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um_2, print_inputs=False)
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um_1, print_inputs=False) 
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um_3, print_inputs=False)

    def test_error_raised_when_input_shape_is_wrong(self):
        # Check number of elements in uv vector different from number of elements in xyz vector 
        with self.assertRaises(ValueError):    
            FitPlane.from_template_centers([[0,1],[0,2],[0,3]],[[0,1,0],[0,2,0]])

        # Check number of elements in uv vector is not two
        with self.assertRaises(ValueError):
            FitPlane.from_template_centers([[0,1],[0,2],[0,3]],[[0,1],[0,2],[0,3]])

        # Check number of elements in xyz vector is not three
        with self.assertRaises(ValueError):
            FitPlane.from_template_centers([[0,1,0],[0,2,0],[0,3,0]],[[0,1,0],[0,2,0],[0,3,0]])

    def test_fit_with_constrains(self):
        uv = [[0,0],[1,0],[0,1]]
        xyz = [[0,0,0],[1,0,0],[0,1,0]]

        n = np.array([0,0.1,0.8])
        n = n / np.linalg.norm(n)

        # Make sure plane fitted norm fits the desired direction
        fp_n = FitPlane.from_template_centers(uv,xyz, forced_plane_normal = n)
        self.assertAlmostEqual(np.dot(fp_n.normal_direction(),n),1, places=1)

        # Compare u and v to the un-forced version
        fp = FitPlane.from_template_centers(uv,xyz)
        self.assertAlmostEqual(fp_n.u_norm_mm(),fp.u_norm_mm(), places=1)
        self.assertAlmostEqual(fp_n.v_norm_mm(),fp.v_norm_mm(), places=1)
