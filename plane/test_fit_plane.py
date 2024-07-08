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

    def test_main_fit_function_runs(self):
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

    def test_translation_on_image_affine(self):
      # Test how well image is transformed using a translation transformation
      dest_image_points = np.array([[p[0]+1,p[1]+2] for p in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      # Create a dummy image and transform it
      source_image = np.zeros((4,6,3))
      source_image[0,0] =  (1,0,0) # Point at the origin (red)
      source_image[2,1] = (0,1,0) # Point  (green)
      source_image[0,3] = (0,0,1) # Point  (blue)
      dest_image = fp.transform_image(source_image)

      # Test the image at the destimation, are pixel values make sense?
      self.assertAlmostEqual(dest_image[2,1,0], 1) # Red visible
      self.assertAlmostEqual(dest_image[2,4,2], 1) # Blue visible
      assert not np.any(dest_image[:,:,1] == 1) # The green point should be out of frame
    
    def test_translation_on_image_quadratic(self):
      # Test how well image is transformed using a translation transformation
      dest_image_points = np.array([[p[0]+1,p[1]+2] for p in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, order=2)

      # Create a dummy image and transform it
      source_image = np.zeros((4,6,3))
      source_image[0,0] =  (1,0,0) # Point at the origin (red)
      source_image[2,1] = (0,1,0) # Point  (green)
      source_image[0,3] = (0,0,1) # Point  (blue)
      dest_image = fp.transform_image(source_image)

      # Test the image at the destimation, are pixel values make sense?
      self.assertAlmostEqual(dest_image[2,1,0], 1) # Red visible
      self.assertAlmostEqual(dest_image[2,4,2], 1) # Blue visible
      assert not np.any(dest_image[:,:,1] == 1) # The green point should be out of frame

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

    def test_rotation_45_affine(self):
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

    def test_rotation_45_quadratic(self):
      # Create array of rotated points
      dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, 2)

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
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, 2)

      # Create a dummy image and transofrm it
      source_image = np.zeros((100,100,3))
      source_image[45:55, 45:55, 0] = 1 # red point
      source_image[90:99, 0:10, 1] = 1 # green point
      source_image[0:10, 90:99, 2] = 1 # blue point
      transformed_image = fp.transform_image(source_image)

      # Plot the transform image
      fig, ax=plt.subplots(1,2)
      ax[0].imshow(source_image)
      ax[1].imshow(transformed_image)
      ax[0].set_title("Source Image")
      ax[1].set_title("Rotated 45 Degrees")
      ax[0].invert_yaxis()
      ax[1].invert_yaxis()
      fig.savefig("test_rotation_45.png")

    def test_transform_image_rotation_translation_affine(self):
      # Test how well image is transformed using a rotation and translation transformation
      rotated_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      dest_image_points = np.array([[p[0]+2,p[1]+4] for p in rotated_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

      # Create a dummy image and transofrm it
      source_image = np.zeros((12,12,3))
      source_image[0,0] =  (1,0,0) # Point at the origin (red)
      source_image[10,0] = (0,1,0) # Point along the Y axis (green)
      source_image[0,10] = (0,0,1) # Point along the X axis (blue)
      dest_image = fp.transform_image(source_image)

      # Test the image at the destimation, are pixel values make sense?
      self.assertAlmostEqual(dest_image[4,2,0], 1) # Point at the origin (red) should have experienced translation only
      self.assertAlmostEqual(dest_image[7+4,7+2,2], 1, places=0) # Pont along the X axis (blue) should have translated and rotated
      assert not np.any(dest_image[:,:,1] == 1) # Point along the Y axis (green) should have moved out of view

    def test_transform_image_rotation_translation_quadratic(self):
      # Test how well image is transformed using a rotation and translation transformation
      rotated_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
      dest_image_points = np.array([[p[0]+2,p[1]+4] for p in rotated_points])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points,2)

      # Create a dummy image and transofrm it
      source_image = np.zeros((12,12,3))
      source_image[0,0] =  (1,0,0) # Point at the origin (red)
      source_image[10,0] = (0,1,0) # Point along the Y axis (green)
      source_image[0,10] = (0,0,1) # Point along the X axis (blue)
      dest_image = fp.transform_image(source_image)

      # Test the image at the destimation, are pixel values make sense?
      self.assertAlmostEqual(dest_image[4,2,0], 1) # Point at the origin (red) should have experienced translation only
      self.assertAlmostEqual(dest_image[7+4,7+2,2], 1, places=0) # Pont along the X axis (blue) should have translated and rotated
      assert not np.any(dest_image[:,:,1] == 1) # Point along the Y axis (green) should have moved out of view

    def test_image_size_specified(self):
      # Create a dummy transformation
      fp=FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, self.source_image_points)

      # Create a dummy image and transofrm it
      source_image = np.zeros((12,12,3))
      dest_image = fp.transform_image(source_image, dest_image_shape=(100,200))

      assert dest_image.shape[0] == 100
      assert dest_image.shape[1] == 200

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
      # Build a random transformation
      source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])
      dest_image_points = np.array([[25,65], [16,13], [70,15], [22,18], [32,46], [40,38]])
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points,2)

      results = []
      for point in source_image_points:
        results.append(fp.transform_point(point))

      for i in range(len(dest_image_points)):
        for j in range(0,2):
          self.assertAlmostEqual(results[i][j], dest_image_points[i][j], places=3)    

    def test_anchor_point_mapping_and_image(self):
      # Test combination of translation, rotation, scaling. Define transofrmation
      source_image_points = [[85,22], [68,43], [88,75], [114,111], [113,166], [76,143]]
      dest_image_points = [[55,142], [71,156], [99,142], [127,123], [170,127], [150,154]]
      fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points,2)

      # Assert that the source_image_points map to dest_image_points 
      transformed_points = []
      for point in source_image_points:
        transformed_points.append(fp.transform_point(point))
      for i in range(len(dest_image_points)):
        for j in range(0,2):
          self.assertAlmostEqual(transformed_points[i][j], dest_image_points[i][j], places=3)

      # Convert arrays to numpy
      transformed_points = np.array(transformed_points)
      source_image_points = np.array(source_image_points)
      dest_image_points = np.array(dest_image_points)

      # Load test images and transform source
      source_image = cv.cvtColor(cv.imread("plane/test_vectors/source.jpg"), cv.COLOR_BGR2RGB)
      dest_image = cv.cvtColor(cv.imread("plane/test_vectors/dest.jpg"), cv.COLOR_BGR2RGB)
      transformed_image = fp.transform_image(source_image)

      # Display results to user
      fig,ax=plt.subplots(1,3)
      ax[0].imshow(source_image)
      ax[1].imshow(transformed_image)
      ax[2].imshow(dest_image)
      ax[0].set_title("Source")
      ax[1].set_title("Transformed")
      ax[2].set_title("Dest")
      ax[0].scatter(source_image_points[:,0], source_image_points[:,1])
      ax[1].scatter(transformed_points[:,0], transformed_points[:,1])
      ax[2].scatter(dest_image_points[:,0],dest_image_points[:,1])
      fig.savefig("test_anchor_points_image.png")

    def _resize_dest(self, source_image, dest_image, dest_points=None):
      # TODO obsolete- resize source NOT dest
      # Resize dest image to be similar dimensions to source image, keeping proportions
      source_y, source_x, _ = source_image.shape
      dest_y, dest_x, _ = dest_image.shape
      original_y, original_x = dest_y, dest_x
      dest_ratio = dest_x/dest_y

      if dest_y > source_y:
        dest_image = cv.resize(dest_image, (int(source_y), int(dest_ratio * source_y)))
        dest_y, dest_x, _ = dest_image.shape
      if dest_x > source_x:
        dest_image = cv.resize(dest_image, (int(1/dest_ratio * source_x), int(source_x)))
        dest_y, dest_x, _ = dest_image.shape

      if np.any(dest_points):
        # Find factor of size reduction
        scaling_factor_y = dest_y / original_y
        scaling_factor_x = dest_x / original_x
        assert abs(scaling_factor_y - scaling_factor_x) < 0.01

        # Rescale dest_points to new positions within dest image
        dest_points = np.array([[round(p[0] * scaling_factor_y), round(p[1] * scaling_factor_x)] for p in dest_points])

      return dest_image, dest_points


if __name__ == '__main__':
  unittest.main()
