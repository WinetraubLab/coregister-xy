import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_template import FitTemplate

class TestFitTemplate(unittest.TestCase):

    def setUp(self):
      self.tk_filepath = "plane/test_vectors/align-1.xml"
      self.l_filepath = "plane/test_vectors/landmarks.xml"
      
    def test_main_function_runs(self):
      FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath)

    def test_parse_data(self):
      # Real saved TrakEM2 XML project files, with original numbers replaced for convenience
      tk_data = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath)
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
      tk_data = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath)
      tk_data.set_M(np.eye(3))
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
      err = tk_data.find_transformation_error()
      self.assertAlmostEqual(err, 2)
    
    def test_compute_error(self):
      test_project = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.set_M(np.eye(3))
      test_project.source_points = np.array([[1,2], [3,4]])
      test_project.dest_points = np.array([[3, 2], [3,6]])
      err = test_project.find_transformation_error()
      self.assertAlmostEqual(err, 2)

      test_project = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.set_M(np.array([[1,0,2], [0,1,5], [0,0,1]]))
      test_project.source_points = np.array([[1,2], [3,4]])
      test_project.dest_points = np.array([[3, 7], [5,9]])
      err = test_project.find_transformation_error()
      self.assertAlmostEqual(err, 0)
    
    def test_physical_POLAR(self):
      deg = 10
      theta = np.deg2rad(deg)
      R2 = np.array([
          [np.cos(theta), -np.sin(theta), 0],
          [np.sin(theta), np.cos(theta), 0],
          [0, 0, 1]
      ])
      scale = 3
      S = np.array([
          [scale, 0, 0],
          [0, scale, 0],
          [0, 0, 1]
      ])
      H = np.array([
         [2,0,0],
         [0,0.5,0],
         [0,0,1]
      ])

      test_project = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.set_M(np.eye(3) @ R2 @ H @ S)
      s = test_project.scale
      r = test_project.theta_deg
      m = test_project.shear_magnitude
      v = test_project.shear_vector
      self.assertAlmostEqual(r, deg)
      self.assertAlmostEqual(s, scale)
      self.assertAlmostEqual(m, 1)
      self.assertAlmostEqual(v[0], 1)
    
    def test_polar_real_matrix(self):
      test_project = FitTemplate.from_imagej_xml(self.tk_filepath, 8, 11, self.l_filepath, True)
      test_project.set_M(np.array([[ 1.84403066, -2.40635941e-01 , 4.98432352e+02],[ 2.56386722e-01 , 2.04612135e+00 , 2.72037605e+03],[ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]]))
      s = test_project.scale
      r = test_project.theta_deg
      m = test_project.shear_magnitude
      v = test_project.shear_vector

if __name__ == '__main__':
  unittest.main()
