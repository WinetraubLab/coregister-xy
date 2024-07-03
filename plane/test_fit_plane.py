import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane
import matplotlib.pyplot as plt
import cv2 as cv

class TestFitPlane(unittest.TestCase):

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

    def test_rotation_45_and_image(self):
      # Create array of rotated points
      dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
      
      source_image = np.zeros((100,100,3))
      source_image[45:55, 45:55, 0] = 1
      source_image[90:99, 0:10, 1] = 1 # green
      source_image[0:10, 90:99, 2] = 1 # blue
      

      transformed_image = fp.transform_image(source_image)

      fig, ax=plt.subplots(1,2)
      ax[0].imshow(source_image)
      ax[1].imshow(transformed_image)
      ax[0].set_title("Source Image")
      ax[1].set_title("Rotated 45 Degrees")
      ax[0].invert_yaxis()
      ax[1].invert_yaxis()
      # fig.show()
      fig.savefig("test_rotation_45.png")


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

    def test_anchor_point_mapping(self):
      # Test combination of translation, rotation, scaling
      source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])
      dest_image_points = np.array([[25,65], [16,13], [70,15], [22,18], [32,46], [40,38]])

      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points)

      results = []
      for point in source_image_points:
        results.append(fp.transform_point(point))

      for i in range(len(dest_image_points)):
        for j in range(0,2):
          self.assertAlmostEqual(results[i][j], dest_image_points[i][j], places=3)

    def test_transform_image(self):
      # Test image transformation
      source_image = np.zeros((12,12,3))
      source_image[0,0] = (1,0,0)
      source_image[10,0] = (0,1,0) # y axis
      source_image[0,10] = (0,0,1) # x axis

      rotated_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      dest_image_points = np.array([[p[0]+2,p[1]+4] for p in rotated_points])

      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      dest_image = fp.transform_image(source_image)
      self.assertAlmostEqual(dest_image[4,2,0], 1) # 0,0
      self.assertAlmostEqual(dest_image[7+4,7+2,2], 1) # x axis originating point
      
      assert not np.any(dest_image[:,:,1] == 1) # y axis originating point

    def test_anchor_point_mapping_and_image(self):
      # Test combination of translation, rotation, scaling
      source_image_points = [[85,22], [68,43], [88,75], [114,111], [113,166], [76,143]]
      dest_image_points = [[55,142], [71,156], [99,142], [127,123], [170,127], [150,154]]

      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points)

      results = []
      for point in source_image_points:
        results.append(fp.transform_point(point))

      for i in range(len(dest_image_points)):
        for j in range(0,2):
          self.assertAlmostEqual(results[i][j], dest_image_points[i][j], places=3)

      # Load images
      source_image = cv.cvtColor(cv.imread("plane/test_vectors/source.jpg"), cv.COLOR_BGR2RGB)
      target_image = cv.cvtColor(cv.imread("plane/test_vectors/target.jpg"), cv.COLOR_BGR2RGB)

      transformed_image = fp.transform_image(source_image)

      source_image_points = np.array(source_image_points)
      dest_image_points = np.array(dest_image_points)

      transformed_points = []
      for point in source_image_points:
        transformed_points.append(fp.transform_point(point))

      transformed_points = np.array(transformed_points)

      fig,ax=plt.subplots(1,3)
      ax[0].imshow(source_image)
      ax[1].imshow(transformed_image)
      ax[2].imshow(target_image)
      ax[0].set_title("Source")
      ax[1].set_title("Transformed")
      ax[2].set_title("Target")
      ax[0].scatter(source_image_points[:,0], source_image_points[:,1])
      ax[1].scatter(transformed_points[:,0], transformed_points[:,1])
      ax[2].scatter(dest_image_points[:,0],dest_image_points[:,1])
      fig.savefig("test_anchor_points_image.png")

    


if __name__ == '__main__':
  unittest.main()
