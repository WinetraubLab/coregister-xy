import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_multi_plane import FitMultiPlane
from plane.parse_xml import ParseXML

class TestFitMultiPlane(unittest.TestCase):

    def setUp(self):
      tk_filepath = "plane/test_vectors/align3.xml"
      fp1 = ParseXML.from_imagej_xml(tk_filepath, 8, 11, None)
      fp2 = ParseXML.from_imagej_xml(tk_filepath, 8, 14, None)
      fp3 = ParseXML.from_imagej_xml(tk_filepath, 8, 17, None)
      self.fp_list = [fp1, fp2, fp3]
      self.real_centers_list = [[0,1000], [1000, 1000], [0, 0]]

    def test_main_function_runs(self):
       FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.real_centers_list, template_size=401, um_per_pixel=2)