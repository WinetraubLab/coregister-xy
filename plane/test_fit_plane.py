import math
import numpy as np
import numpy.testing as npt
import unittest
from fit_plane import FitPlane

class TestFitPlane(unittest.TestCase):

    # Define default pattern
    def setUp(self):
      self.source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])

    def test_main_function_runs(self):
      FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, self.source_image_points)

    def test_translation(self):
      dest_image_points = np.array([[p[0]+10,p[1]+5] for p in self.source_image_points])

      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
      
      # Apply the transformation on the source points and make sure that it matches destination points
      result1 = fp.transform_point([0,1])
      self.assertAlmostEqual(result1[0], 10)
      self.assertAlmostEqual(result1[1], 6)

      result2 = fp.transform_point([1,0])
      self.assertAlmostEqual(result2[0], 11)
      self.assertAlmostEqual(result2[1], 5)

    def _rotate_point(self,x,y,angle):
        # Rotates a point by specified degree angle
        angle_radians = math.radians(angle)
        cos_theta = math.cos(angle_radians)
        sin_theta = math.sin(angle_radians)
        x_new = x * cos_theta - y * sin_theta
        y_new = x * sin_theta + y * cos_theta
        return x_new, y_new
    
    def test_rotation_90(self):
      # Create array of rotated points

      dest_image_points = np.array([self._rotate_point(x,y,90) for [x,y] in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      # Apply the transformation on the source points
      result1 = fp.transform_point([0,1]) 
      result2 = fp.transform_point([1,0]) 

      self.assertAlmostEqual(result1[0], -1)
      self.assertAlmostEqual(result1[1], 0)
      self.assertAlmostEqual(result2[0], 0)
      self.assertAlmostEqual(result2[1], 1)

    def test_rotation_45(self):
      # Create array of rotated points

      dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      # Apply the transformation on the source points
      result1 = fp.transform_point([0,1])
      result2 = fp.transform_point([1,0]) 

      self.assertAlmostEqual(result1[0], -1 * 1/math.sqrt(2))
      self.assertAlmostEqual(result1[1], 1/math.sqrt(2))
      self.assertAlmostEqual(result2[0], 1/math.sqrt(2))
      self.assertAlmostEqual(result2[1], 1/math.sqrt(2))

    def test_scaling(self):
      # Scale by factor of 2 on x, factor of 4 on y
      dest_image_points = np.array([[p[0]*2,p[1]*4] for p in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      result1 = fp.transform_point([0,1])
      result2 = fp.transform_point([1,0])
      result3 = fp.transform_point([0,0])

      self.assertAlmostEqual(result1[0], 0)
      self.assertAlmostEqual(result1[1], 4)
      self.assertAlmostEqual(result2[0], 2)
      self.assertAlmostEqual(result2[1], 0)
      self.assertAlmostEqual(result3[0], 0)
      self.assertAlmostEqual(result3[1], 0)




if __name__ == '__main__':
  unittest.main()
