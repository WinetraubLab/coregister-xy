import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane
from plane.parse_xml import ParseXML
import matplotlib.pyplot as plt
import cv2 as cv

class TestFitPlane(unittest.TestCase):

    def setUp(self):
      self.source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])

    # def _rotate_point(self,x,y,angle):
    #     # Rotates a point by specified degree angle
    #     angle_radians = math.radians(angle)
    #     cos_theta = math.cos(angle_radians)
    #     sin_theta = math.sin(angle_radians)
    #     x_new = x * cos_theta - y * sin_theta
    #     y_new = x * sin_theta + y * cos_theta
    #     return x_new, y_new
      
    # def test_main_fit_function_runs(self):
    #   FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, self.source_image_points)

    # def test_translation(self):
    #   dest_image_points = np.array([[p[0]+10,p[1]+5] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
      
    #   # Apply the transformation on the source points and make sure that it matches destination points
    #   result1 = fp.transform_point([0,1])
    #   self.assertAlmostEqual(result1[0], 10)
    #   self.assertAlmostEqual(result1[1], 6)

    #   result2 = fp.transform_point([1,0])
    #   self.assertAlmostEqual(result2[0], 11)
    #   self.assertAlmostEqual(result2[1], 5)

    # def test_translation_on_image_affine(self):
    #   # Test how well image is transformed using a translation transformation
    #   dest_image_points = np.array([[p[0]+1,p[1]+2] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   # Create a dummy image and transform it
    #   source_image = np.zeros((4,6,3))
    #   source_image[0,0] =  (1,0,0) # Point at the origin (red)
    #   source_image[2,1] = (0,1,0) # Point  (green)
    #   source_image[0,3] = (0,0,1) # Point  (blue)
    #   dest_image = fp.transform_image(source_image)

    #   # Test the image at the destination, are pixel values make sense?
    #   self.assertAlmostEqual(dest_image[2,1,0], 1) # Red visible
    #   self.assertAlmostEqual(dest_image[2,4,2], 1) # Blue visible
    #   assert not np.any(dest_image[:,:,1] == 1) # The green point should be out of frame
    
    # def test_translation_on_image_quadratic(self):
    #   # Test how well image is transformed using a translation transformation
    #   dest_image_points = np.array([[p[0]+1,p[1]+2] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, order=2)

    #   # Create a dummy image and transform it
    #   source_image = np.zeros((4,6,3))
    #   source_image[0,0] =  (1,0,0) # Point at the origin (red)
    #   source_image[2,1] = (0,1,0) # Point  (green)
    #   source_image[0,3] = (0,0,1) # Point  (blue)
    #   dest_image = fp.transform_image(source_image)

    #   # Test the image at the destimation, are pixel values make sense?
    #   self.assertAlmostEqual(dest_image[2,1,0], 1) # Red visible
    #   self.assertAlmostEqual(dest_image[2,4,2], 1) # Blue visible
    #   assert not np.any(dest_image[:,:,1] == 1) # The green point should be out of frame
    
    # def test_rotation_90(self):
    #   # Create array of rotated points
    #   dest_image_points = np.array([self._rotate_point(x,y,90) for [x,y] in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   # Apply the transformation on the source points
    #   result1 = fp.transform_point([0,1]) 
    #   result2 = fp.transform_point([1,0]) 

    #   self.assertAlmostEqual(result1[0], -1)
    #   self.assertAlmostEqual(result1[1], 0)
    #   self.assertAlmostEqual(result2[0], 0)
    #   self.assertAlmostEqual(result2[1], 1)

    # def test_rotation_45_affine(self):
    #   # Create array of rotated points
    #   dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   # Apply the transformation on the source points
    #   result1 = fp.transform_point([0,1])
    #   result2 = fp.transform_point([1,0]) 

    #   self.assertAlmostEqual(result1[0], -1 * 1/math.sqrt(2))
    #   self.assertAlmostEqual(result1[1], 1/math.sqrt(2))
    #   self.assertAlmostEqual(result2[0], 1/math.sqrt(2))
    #   self.assertAlmostEqual(result2[1], 1/math.sqrt(2))

    # def test_rotation_45_quadratic(self):
    #   # Create array of rotated points
    #   dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, 2)

    #   # Apply the transformation on the source points
    #   result1 = fp.transform_point([0,1])
    #   result2 = fp.transform_point([1,0]) 

    #   self.assertAlmostEqual(result1[0], -1 * 1/math.sqrt(2))
    #   self.assertAlmostEqual(result1[1], 1/math.sqrt(2))
    #   self.assertAlmostEqual(result2[0], 1/math.sqrt(2))
    #   self.assertAlmostEqual(result2[1], 1/math.sqrt(2))

    # def test_rotation_45_and_image(self):
    #   # Create array of rotated points
    #   dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points, 2)

    #   # Create a dummy image and transofrm it
    #   source_image = np.zeros((100,100,3))
    #   source_image[45:55, 45:55, 0] = 1 # red point
    #   source_image[90:99, 0:10, 1] = 1 # green point
    #   source_image[0:10, 90:99, 2] = 1 # blue point
    #   transformed_image = fp.transform_image(source_image)

    #   # Plot the transform image
    #   fig, ax=plt.subplots(1,2)
    #   ax[0].imshow(source_image)
    #   ax[1].imshow(transformed_image)
    #   ax[0].set_title("Source Image")
    #   ax[1].set_title("Rotated 45 Degrees")
    #   ax[0].invert_yaxis()
    #   ax[1].invert_yaxis()
    #   fig.savefig("test_rotation_45.png")

    # def test_transform_image_rotation_translation_affine(self):
    #   # Test how well image is transformed using a rotation and translation transformation
    #   rotated_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   dest_image_points = np.array([[p[0]+2,p[1]+4] for p in rotated_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   # Create a dummy image and transofrm it
    #   source_image = np.zeros((12,12,3))
    #   source_image[0,0] =  (1,0,0) # Point at the origin (red)
    #   source_image[10,0] = (0,1,0) # Point along the Y axis (green)
    #   source_image[0,10] = (0,0,1) # Point along the X axis (blue)
    #   dest_image = fp.transform_image(source_image)

    #   # Test the image at the destimation, are pixel values make sense?
    #   self.assertAlmostEqual(dest_image[4,2,0], 1) # Point at the origin (red) should have experienced translation only
    #   self.assertAlmostEqual(dest_image[7+4,7+2,2], 1, places=0) # Pont along the X axis (blue) should have translated and rotated
    #   assert not np.any(dest_image[:,:,1] == 1) # Point along the Y axis (green) should have moved out of view

    # def test_transform_image_rotation_translation_quadratic(self):
    #   # Test how well image is transformed using a rotation and translation transformation
    #   rotated_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   dest_image_points = np.array([[p[0]+2,p[1]+4] for p in rotated_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points,2)

    #   # Create a dummy image and transofrm it
    #   source_image = np.zeros((12,12,3))
    #   source_image[0,0] =  (1,0,0) # Point at the origin (red)
    #   source_image[10,0] = (0,1,0) # Point along the Y axis (green)
    #   source_image[0,10] = (0,0,1) # Point along the X axis (blue)
    #   dest_image = fp.transform_image(source_image)

    #   # Test the image at the destimation, are pixel values make sense?
    #   self.assertAlmostEqual(dest_image[4,2,0], 1) # Point at the origin (red) should have experienced translation only
    #   self.assertAlmostEqual(dest_image[7+4,7+2,2], 1, places=0) # Pont along the X axis (blue) should have translated and rotated
    #   assert not np.any(dest_image[:,:,1] == 1) # Point along the Y axis (green) should have moved out of view

    # def test_image_size_specified(self):
    #   # Create a dummy transformation
    #   fp=FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, self.source_image_points)

    #   # Create a dummy image and transofrm it
    #   source_image = np.zeros((12,12,3))
    #   dest_image = fp.transform_image(source_image, dest_image_shape=(100,200))

    #   assert dest_image.shape[0] == 100
    #   assert dest_image.shape[1] == 200

    # def test_scaling(self):
    #   # Scale by factor of 2 on x, factor of 4 on y
    #   dest_image_points = np.array([[p[0]*2,p[1]*4] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   result1 = fp.transform_point([0,1])
    #   result2 = fp.transform_point([1,0])
    #   result3 = fp.transform_point([0,0])

    #   self.assertAlmostEqual(result1[0], 0)
    #   self.assertAlmostEqual(result1[1], 4)
    #   self.assertAlmostEqual(result2[0], 2)
    #   self.assertAlmostEqual(result2[1], 0)
    #   self.assertAlmostEqual(result3[0], 0)
    #   self.assertAlmostEqual(result3[1], 0)

    # def test_anchor_point_mapping(self):
    #   # Build a random transformation
    #   source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])
    #   dest_image_points = np.array([[25,65], [16,13], [70,15], [22,18], [32,46], [40,38]])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points,2)

    #   results = []
    #   for point in source_image_points:
    #     results.append(fp.transform_point(point))

    #   for i in range(len(dest_image_points)):
    #     for j in range(0,2):
    #       self.assertAlmostEqual(results[i][j], dest_image_points[i][j], places=3)    
    
    # def test_reverse_transformation(self):
    #   # Build a random transformation
    #   source_image_points = np.array([ [20,60], [20, 10], [60, 10], [30, 20], [30, 45], [45, 45]])
    #   dest_image_points = np.array([[25,65], [16,13], [70,15], [22,18], [32,46], [40,38]])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points,2)

    #   transformed = []
    #   for point in source_image_points:
    #     transformed.append(fp.transform_point(point))
      
    #   # Use M_rev to reverse transform the points
    #   cycle = []
    #   for point in transformed:
    #     cycle.append(fp.transform_point(point, True))

    #   # Check that the points are back to their starting position
    #   for p in range(0, len(source_image_points)):
    #     for i in range(2):
    #       self.assertAlmostEqual(cycle[p][i], source_image_points[p][i], places=3)

    # def test_anchor_point_mapping_and_image(self):
    #   # Test combination of translation, rotation, scaling. Define transofrmation
    #   source_image_points = [[85,22], [68,43], [88,75], [114,111], [113,166], [76,143]]
    #   dest_image_points = [[55,142], [71,156], [99,142], [127,123], [170,127], [150,154]]
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points,2)

    #   # Assert that the source_image_points map to dest_image_points 
    #   transformed_points = []
    #   for point in source_image_points:
    #     transformed_points.append(fp.transform_point(point))
    #   for i in range(len(dest_image_points)):
    #     for j in range(0,2):
    #       self.assertAlmostEqual(transformed_points[i][j], dest_image_points[i][j], places=3)

    #   # Convert arrays to numpy
    #   transformed_points = np.array(transformed_points)
    #   source_image_points = np.array(source_image_points)
    #   dest_image_points = np.array(dest_image_points)

    #   # Load test images and transform source
    #   source_image = cv.cvtColor(cv.imread("plane/test_vectors/source.jpg"), cv.COLOR_BGR2RGB)
    #   dest_image = cv.cvtColor(cv.imread("plane/test_vectors/dest.jpg"), cv.COLOR_BGR2RGB)
    #   transformed_image = fp.transform_image(source_image)

    #   # Display results to user
    #   fig,ax=plt.subplots(1,3)
    #   ax[0].imshow(source_image)
    #   ax[1].imshow(transformed_image)
    #   ax[2].imshow(dest_image)
    #   ax[0].set_title("Source")
    #   ax[1].set_title("Transformed")
    #   ax[2].set_title("Dest")
    #   ax[0].scatter(source_image_points[:,0], source_image_points[:,1])
    #   ax[1].scatter(transformed_points[:,0], transformed_points[:,1])
    #   ax[2].scatter(dest_image_points[:,0],dest_image_points[:,1])
    #   fig.savefig("test_anchor_points_image.png")
    
    # def test_scale_source(self):
    #   small_source = cv.cvtColor(cv.imread("plane/test_vectors/source.jpg"), cv.COLOR_BGR2RGB)
    #   large_dest = cv.cvtColor(cv.imread("plane/test_vectors/large.jpg"), cv.COLOR_BGR2RGB)

    #   dest_image_points = np.array([[p[0]*2.5,p[1]*2.5] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)

    #   transformed = fp.transform_image(small_source, large_dest.shape)

    #   assert transformed.shape == large_dest.shape

    #   # Display results to user
    #   fig,ax=plt.subplots(1,3)
    #   ax[0].imshow(small_source)
    #   ax[1].imshow(transformed)
    #   ax[2].imshow(large_dest)
    #   ax[0].set_title("Source")
    #   ax[1].set_title("Transformed")
    #   ax[2].set_title("Dest")
    #   fig.savefig("test_scale_source.png")

    # def test_shuffle_points(self):
    #   # 3 points for affine can be used with np.linalg.solve
    #   source_image_points = np.array([ [20,60], [90, 20], [0, 45]])
    #   dest_image_points = np.array([[25,65], [22,15], [32,80]])

    #   # Shuffle the points to create a different order but maintain pairings
    #   indices = np.random.permutation(source_image_points.shape[0])
    #   shuffled_source = source_image_points[indices]
    #   shuffled_dest = dest_image_points[indices]
    #   tolerance=1e-5

    #   # Check that shuffling point pairs does not affect M
    #   fp_original = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points, dest_image_points)      
    #   fp_shuffled = FitPlane.from_fitting_points_between_fluorescence_image_and_template(shuffled_source, shuffled_dest)      
    #   assert np.allclose(fp_original.M, fp_shuffled.M, atol=tolerance)

    # def test_adding_points_changes_transform(self):
    #   # Build a random transformation by picking random points
    #   source_image_points = np.array([[20,60], [30, 10], [60, 10], [90, 20], [0, 45], [45, 45]])
    #   dest_image_points = np.array([[25,65], [16,54], [70,14], [22,15], [32,80], [40,28]])

    #   tolerance=1e-5

    #   # Using sub-set of points, build a plane fit 
    #   fp3 = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points[0:3], dest_image_points[0:3])
    #   fp4 = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points[0:4], dest_image_points[0:4])
    #   fp5 = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points[0:5], dest_image_points[0:5])      
    #   fp6 = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_image_points[0:6], dest_image_points[0:6])      

    #   # Because all the points are chosen in random, we expect the different fits to have different plane parameters.
    #   assert not np.allclose(fp3.M, fp4.M, atol=tolerance)
    #   assert not np.allclose(fp3.M, fp5.M, atol=tolerance)
    #   assert not np.allclose(fp3.M, fp6.M, atol=tolerance)
    #   assert not np.allclose(fp4.M, fp5.M, atol=tolerance)
    #   assert not np.allclose(fp4.M, fp6.M, atol=tolerance)
    #   assert not np.allclose(fp5.M, fp6.M, atol=tolerance)
    
    # def test_compute_physical_translate(self):
    #   dest_image_points = np.array([[p[0]+1,p[1]+2] for p in self.source_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
    #   (tx,ty), theta, sx, sy, shear = fp.compute_physical_params()
    #   self.assertAlmostEqual(tx, 1)
    #   self.assertAlmostEqual(ty, 2)
    #   self.assertAlmostEqual(theta, 0)
    #   self.assertAlmostEqual(sx, 1)
    #   self.assertAlmostEqual(sy, 1)
    #   self.assertAlmostEqual(shear, 0)
    
    # def test_compute_physical_rotate_translate(self):
    #   # rotate then translate
    #   dest_image_points = np.array([self._rotate_point(x,y,45) for [x,y] in self.source_image_points])
    #   dest_image_points = np.array([[p[0]+1,p[1]+2] for p in dest_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
    #   (tx,ty), theta, sx, sy, shear = fp.compute_physical_params()
    #   self.assertAlmostEqual(tx, 1)
    #   self.assertAlmostEqual(ty, 2)
    #   self.assertAlmostEqual(theta, 45)
    #   self.assertAlmostEqual(sx, 1)
    #   self.assertAlmostEqual(sy, 1)
    #   self.assertAlmostEqual(shear, 0)

    # def test_compute_physical_scale_rotate_translate(self):
    #   # scale then rotate then translate
    #   dest_image_points = np.array([[p[0]*2,p[1]*3] for p in self.source_image_points])
    #   dest_image_points = np.array([self._rotate_point(x,y,30) for [x,y] in dest_image_points])
    #   dest_image_points = np.array([[p[0]+1,p[1]+2] for p in dest_image_points])
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
    #   (tx,ty), theta, sx, sy, shear = fp.compute_physical_params()
    #   self.assertAlmostEqual(tx, 1, places=3)
    #   self.assertAlmostEqual(ty, 2, places=3)
    #   self.assertAlmostEqual(theta, 30, places=3)
    #   self.assertAlmostEqual(sx, 2, places=3)
    #   self.assertAlmostEqual(sy, 3, places=3)
    #   self.assertAlmostEqual(shear, 0)

    # def test_compute_physical_scale_rotate_translate(self):
    #   # scale then rotate then translate
    #   # construct transformations
    #   S = np.array([
    #       [2, 0, 0],
    #       [0, 3, 0],
    #       [0, 0, 1]
    #   ])
    #   shear = 0
    #   H = np.array([
    #       [1, shear, 0],
    #       [0, 1, 0],
    #       [0, 0, 1]
    #   ])
    #   theta = np.deg2rad(30)
    #   R = np.array([
    #       [np.cos(theta), -np.sin(theta), 0],
    #       [np.sin(theta), np.cos(theta), 0],
    #       [0, 0, 1]
    #   ])

    #   T = np.array([
    #       [1, 0, 1],
    #       [0, 1, 2],
    #       [0, 0, 1]
    #   ])

    #   M = T @ R @ H @ S
    #   source_image_points = np.hstack([self.source_image_points, np.ones((self.source_image_points.shape[0], 1))])
    #   dest_image_points = source_image_points @ M.T
    #   dest_image_points = dest_image_points[:, :2]
    #   fp = FitPlane.from_fitting_points_between_fluorescence_image_and_template(self.source_image_points, dest_image_points)
    #   (tx,ty), theta, sx, sy, shear_x, shear_y = fp.compute_physical_params()
    #   self.assertAlmostEqual(tx, 1, places=3)
    #   self.assertAlmostEqual(ty, 2, places=3)
    #   self.assertAlmostEqual(theta, 30, places=3)
    #   self.assertAlmostEqual(sx, 2, places=3)
    #   self.assertAlmostEqual(sy, 3, places=3)
    #   self.assertAlmostEqual(shear_x, 0)
    #   self.assertAlmostEqual(shear_y, 0)
      
    # def test_shear_on_image(self):
    #    source_points = [[36,56], [159,54], [159,255], [36,256]]
    #    up_points = [[36,45], [159,8], [159,209], [36,246]]
    #    right_points = [[71,56], [194, 54], [165,255], [42,255]]
    #    s45_points = [[46,67], [175,37], [150,246], [33,259]]

    #    fp_control = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_points, source_points)
    #    fp_up = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_points, up_points)
    #    fp_right = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_points, right_points)
    #    fp_45 = FitPlane.from_fitting_points_between_fluorescence_image_and_template(source_points, s45_points)

    #    (tx,ty), theta, sx, sy, shear_dir, shear_mag = fp_control.compute_physical_params()
    #    self.assertAlmostEqual(shear_mag, 0)
    #    (tx,ty), theta, sx, sy, shear_dir, shear_mag = fp_up.compute_physical_params()
    #   #  self.assertAlmostEqual(shear_mag, 0)

    #    print((tx,ty), theta, sx, sy, shear_dir, shear_mag)
    #    print(fp_up.M_rev)

    def test_parse_xml(self):
      # Real saved TrakEM2 XML project files, with original numbers replaced for convenience
      tk_filepath = "plane/test_vectors/align-1.xml"
      l_filepath = "plane/test_vectors/landmarks.xml"
      tk_data = ParseXML.extract_data(tk_filepath, 8, 11, l_filepath)
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

if __name__ == '__main__':
  unittest.main()
