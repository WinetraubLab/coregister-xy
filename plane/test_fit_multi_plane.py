import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane
from plane.fit_template import FitTemplate

class TestFitPlane(unittest.TestCase):

    def setUp(self):
      tk_filepath = "plane/test_vectors/align3.xml"
      fp1 = FitTemplate.from_imagej_xml(tk_filepath, 8, 11, None)
      fp2 = FitTemplate.from_imagej_xml(tk_filepath, 8, 14, None)
      fp3 = FitTemplate.from_imagej_xml(tk_filepath, 8, 17, None)
      fp_z_list = [48, 50, 52]
      self.avg_scale = np.mean([fp1.scale, fp2.scale, fp3.scale])
      self.template_size = 401
      self.templates_px = []
      for i, fp in enumerate([fp1, fp2, fp3]):
        self.templates_px.append((fp.tx + self.template_size/2, fp.ty + self.template_size/2, fp_z_list[i]))
      self.target_centers_list = [[0,1000], [1000, 1000], [0, 0]]

    def test_main_function_runs(self):
       FitPlane.from_aligned_fit_templates(self.templates_px, self.target_centers_list, self.avg_scale, template_size=self.template_size, um_per_pixel=2)