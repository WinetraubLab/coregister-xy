import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane_elastic import FitPlaneElastic
import cv2

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        self.template_center_positions_uv_pix = [[0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        self.template_center_positions_xyz_mm = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]]

    def test_main_function_runs(self):
        FitPlaneElastic.from_points(self.template_center_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        FitPlaneElastic.from_points(self.template_center_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=True)

    # def test_fit_mapping(self):
    #     fp = FitPlaneElastic.from_points(self.template_center_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
