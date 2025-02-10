
import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_tps import FitPlaneTPS

class TestFitPlane(unittest.TestCase):

    def setUp(self):
        self.template_center_positions_uv_pix = [[0,1],[1,0],[1,1]]
        self.template_center_positions_xyz_mm = [[0,1,0],[1,0,0],[1,1,0]]

    def test_main_function_runs(self):
        FitPlaneTPS.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        FitPlaneTPS.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=True)
