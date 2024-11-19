
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
