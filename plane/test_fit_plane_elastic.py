import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane_elastic import FitPlaneElastic
import cv2

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        self.fluorescent_image_points_positions_uv_pix = [[0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        self.template_center_positions_xyz_mm = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]]

    def test_main_function_runs(self):
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=True)
    
    def test_get_xyz_from_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        npt.assert_array_almost_equal(xyz, self.template_center_positions_xyz_mm)

    def test_get_uv_from_xyz(self):
        uv_pix = [[4, 1], [5, 0], [5, 1], [4.5, 0.5]]
        fp = FitPlaneElastic.from_points(uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        uv = fp.get_uv_from_xyz(self.template_center_positions_xyz_mm)
        npt.assert_array_almost_equal(uv, uv_pix)
