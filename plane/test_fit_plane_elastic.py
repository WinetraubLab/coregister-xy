import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane_elastic import FitPlaneElastic
import cv2

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        self.fluorescent_image_points_positions_uv_pix = [[0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        self.template_positions_xyz_mm = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]]

    def test_main_function_runs(self):
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=True)
    
    def test_get_xyz_from_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        # Check that transformation projects uv to xyz 
        npt.assert_array_almost_equal(xyz, self.template_positions_xyz_mm)

    def test_get_uv_from_xyz(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        uv = fp.get_uv_from_xyz(self.template_positions_xyz_mm)
        # Check that transformation projects xyz to uv
        npt.assert_array_almost_equal(uv, self.fluorescent_image_points_positions_uv_pix)

    def test_uv_to_xyz_back_to_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        uv = fp.get_uv_from_xyz(xyz)
        npt.assert_array_almost_equal(uv, self.fluorescent_image_points_positions_uv_pix)
