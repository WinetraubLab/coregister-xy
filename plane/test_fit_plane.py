import math
import numpy as np
import numpy.testing as npt
import unittest

class TestFitPlane(unittest.TestCase):

    # Define default pattern
    def setUp(self):
      self.source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45])
            
    def assertAlmostEqualRelative(self, first, second, rel_tol=1e-3, msg=None):
        if not math.isclose(first, second, rel_tol=rel_tol):
            standard_msg = f'{first} != {second} within {rel_tol} relative tolerance'
            self.fail(self._formatMessage(msg, standard_msg))
                
    def test_main_function_runs(self):
      raise NotImplemented()

    def test_translation(self):
      dest_image_points = np.array([[p[0]+10,p[1]+10] for p in self.source_image_points.toArray()])
      fp = 1  # TBD

      # Apply the transformation on the source points and make sure that it matches destination points

if __name__ == '__main__':
    unittest.main()
