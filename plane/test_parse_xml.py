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

    def test_compute_physical_scale_rotate_translate(self):
      # scale then rotate then translate
      # construct transformations
      S = np.array([
          [2, 0, 0],
          [0, 3, 0],
          [0, 0, 1]
      ])
      H = np.array([
          [1, 2, 0],
          [5, 1, 0],
          [0, 0, 1]
      ])
      theta = np.deg2rad(30)
      R = np.array([
          [np.cos(theta), -np.sin(theta), 0],
          [np.sin(theta), np.cos(theta), 0],
          [0, 0, 1]
      ])

      T = np.array([
          [1, 0, 10],
          [0, 1, 20],
          [0, 0, 1]
      ])

      M = T @ R @ S @ H
      tk_data = ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath)
      tk_data.M = M
      tx, ty, theta, sx, sy, shear = tk_data.compute_physical_params()

      S2 = np.array([
          [sx, 0, 0],
          [0, sy, 0],
          [0, 0, 1]
      ])
      H2 = np.array([
          [1, shear, 0],
          [0, 1, 0],
          [0, 0, 1]
      ])
      theta = np.deg2rad(theta)
      R2 = np.array([
          [np.cos(theta), -np.sin(theta), 0],
          [np.sin(theta), np.cos(theta), 0],
          [0, 0, 1]
      ])

      T2 = np.array([
          [1, 0, tx],
          [0, 1, ty],
          [0, 0, 1]
      ])
      M2 = T2 @ R2 @ S2 @ H2
      
      for row in range(0, M2.shape[0]):
         for col in range(0, M2.shape[1]):
            self.assertAlmostEqual(M2[row,col], M[row, col])

    def test_compute_physical_from_xml(self):
      # scale then rotate then translate
      tk_data = ParseXML.extract_data("plane/test_vectors/trakem-sample.xml", 8, 11, "plane/test_vectors/landmarks-sample.xml")
      tx, ty, theta, sx, sy, shear = tk_data.compute_physical_params()

      S2 = np.array([
          [sx, 0, 0],
          [0, sy, 0],
          [0, 0, 1]
      ])
      H2 = np.array([
          [1, shear, 0],
          [0, 1, 0],
          [0, 0, 1]
      ])
      theta = np.deg2rad(theta)
      R2 = np.array([
          [np.cos(theta), -np.sin(theta), 0],
          [np.sin(theta), np.cos(theta), 0],
          [0, 0, 1]
      ])

      T2 = np.array([
          [1, 0, tx],
          [0, 1, ty],
          [0, 0, 1]
      ])
      M2 = T2 @ R2 @ S2 @ H2
      
      for row in range(0, M2.shape[0]):
         for col in range(0, M2.shape[1]):
            self.assertAlmostEqual(M2[row,col], tk_data.M[row, col])
    
    def test_compute_error(self):
      test_project = ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.M = np.array([[1,0,0], [0,1,0], [0,0,1]])
      test_project.source_points = np.array([[1,2], [3,4]])
      test_project.dest_points = np.array([[3, 2], [3,6]])
      err = test_project.find_transformation_error()
      self.assertAlmostEqual(err, 2)

      test_project.M = np.array([[1,0,2], [0,1,5], [0,0,1]])
      test_project.source_points = np.array([[1,2], [3,4]])
      test_project.dest_points = np.array([[3, 7], [5,9]])
      err = test_project.find_transformation_error()
      self.assertAlmostEqual(err, 0)

    def test_calc_new_scale(self):
      test_project = ParseXML.extract_data(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.M = np.array([[1,0,0], [0,1,0], [0,0,1]])
      self.assertAlmostEqual(test_project.calc_real_scale(1, 0), 1)

      test_project.M = np.array([[2,0,0], [0,3,0], [0,0,1]])
      self.assertAlmostEqual(test_project.calc_real_scale(1, 0), 0.5)
      self.assertAlmostEqual(test_project.calc_real_scale(2, 90), 2/3)

if __name__ == '__main__':
  unittest.main()
