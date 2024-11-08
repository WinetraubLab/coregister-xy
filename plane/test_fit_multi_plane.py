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
      self.fp_list = [fp1, fp2, fp3]
      self.real_centers_list = [[0,1000], [1000, 1000], [0, 0]]

    def test_main_function_runs(self):
       FitPlane.from_aligned_fit_templates(self.fp_list, self.real_centers_list, template_size=401, um_per_pixel=2)