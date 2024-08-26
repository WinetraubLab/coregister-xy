import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.parse_xml import ParseXML
import matplotlib.pyplot as plt
import cv2 as cv

class TestParseXML(unittest.TestCase):

    def setUp(self):
      self.tk_filepath = "plane/test_vectors/align-1.xml"
      self.l_filepath = "plane/test_vectors/landmarks.xml"
      
    def test_main_function_runs(self):
      ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath)

    def test_parse_data(self):
      # Real saved TrakEM2 XML project files, with original numbers replaced for convenience
      tk_data = ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath)
      self.assertAlmostEqual(tk_data.M[0,0], 1)
      self.assertAlmostEqual(tk_data.M[1,0], 2)
      self.assertAlmostEqual(tk_data.M[0,1], 3)
      self.assertAlmostEqual(tk_data.M[0,2], 500)
      self.assertAlmostEqual(tk_data.M[1,2], 600)
      self.assertAlmostEqual(tk_data.source_points[0,0],800)
      self.assertAlmostEqual(tk_data.source_points[0,1],801)
      self.assertAlmostEqual(tk_data.source_points[1,0],80)
      self.assertAlmostEqual(tk_data.source_points[1,1],81)
      self.assertAlmostEqual(tk_data.dest_points[1,0],11)
      self.assertAlmostEqual(tk_data.dest_points[1,1],10)

    def test_find_transformation_error(self):
      tk_data = ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath)
      tk_data.source_points = np.array([
        [1,2],
        [4,5],
        [8,8]
      ])
      # translate y by 2
      tk_data.dest_points = np.array([
        [1,4],
        [6,5],
        [8,10]
      ])
      tk_data.M = np.eye(3)
      err = tk_data.find_transformation_error()
      self.assertAlmostEqual(err, 2)

if __name__ == '__main__':
  unittest.main()
