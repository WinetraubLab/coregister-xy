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
      result = fp.transform_point([0,0])
      self.assertAlmostEqual(result[0], 10)
      self.assertAlmostEqual(result[1], 5)

      # Apply the transformation on the source points and make sure that it matches destination points

if __name__ == '__main__':
  unittest.main()
