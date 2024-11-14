
import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane

class TestFitMultiPlane(unittest.TestCase):

    def setUp(self):
      self.template_center_positions_uv_pix = [[0,1],[1,0],[1,1]]
      self.template_center_positions_xyz_um = [[0,2,10], [2,0,10], [2,2,11]]

    def test_main_function_runs(self):
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um, print_inputs=False)
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um, print_inputs=True)
