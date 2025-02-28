import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane_elastic import FitPlaneElastic
import cv2
from unittest.mock import patch

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        self.fluorescent_image_points_positions_uv_pix = [[0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        self.template_center_positions_xyz_mm = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]]

    def test_main_function_runs(self):
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=True)
    
    def test_get_xyz_from_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        npt.assert_array_almost_equal(xyz, self.template_center_positions_xyz_mm)

    def test_get_uv_from_xyz(self):
        uv_pix = [[4, 1], [5, 0], [5, 1], [4.5, 0.5]]
        fp = FitPlaneElastic.from_points(uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)
        uv = fp.get_uv_from_xyz(self.template_center_positions_xyz_mm)
        npt.assert_array_almost_equal(uv, uv_pix)

    def test_image_to_physical_translations_xy(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300], [100,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0], [1,3,0]] # mm
        fp = FitPlaneElastic.from_points(uv,xyz)
        np.random.seed(42)
        random_image = np.random.randint(0, 256, (300, 100, 3), dtype=np.uint8) # 100 by 300 noise

        # Map that image to the right, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[-0.1,0.9], y_range_mm=[0,3], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],0)
        self.assertAlmostEqual(mapped_image[0,0,1],0)
        self.assertAlmostEqual(mapped_image[0,0,2],0)
        self.assertAlmostEqual(mapped_image[11,0,0],0)
        self.assertGreater(mapped_image[10,30,0],0) # Area which shouldn't be effected

        # Map that image to the left, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0.1,1.1], y_range_mm=[0,3], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,100-5,0],0)
        self.assertAlmostEqual(mapped_image[0,100-5,1],0)
        self.assertAlmostEqual(mapped_image[0,100-5,2],0)
        self.assertAlmostEqual(mapped_image[10,100-5,0],0)
        self.assertGreater(mapped_image[10,100-11,0],0) # Area which shouldn't be effected

        # Map that image down, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[-0.1,2.9], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],0)
        self.assertAlmostEqual(mapped_image[0,0,1],0)
        self.assertAlmostEqual(mapped_image[0,0,2],0)
        self.assertAlmostEqual(mapped_image[0,11,0],0)
        self.assertGreater(mapped_image[30,11,0],0) # Area which shouldn't be effected

        # Map that image up, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[300-5,0,0],0)
        self.assertAlmostEqual(mapped_image[300-5,0,1],0)
        self.assertAlmostEqual(mapped_image[300-5,0,2],0)
        self.assertAlmostEqual(mapped_image[300-5,10,0],0)
        self.assertGreater(mapped_image[300-11,10,0],0) # Area which shouldn't be effected

        # Map that image up, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[300-5,0,0],0)
        self.assertAlmostEqual(mapped_image[300-5,0,1],0)
        self.assertAlmostEqual(mapped_image[300-5,0,2],0)
        self.assertAlmostEqual(mapped_image[300-5,10,0],0)
        self.assertGreater(mapped_image[300-11,10,0],0) # Area which shouldn't be effected

        # Shift to arbitrary point
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0.2,1.2], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        # self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],random_image[10,20,0])

    def test_image_to_physical_output_size(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0]] # mm
        fp = FitPlaneElastic.from_points(uv,xyz)
        random_image = np.random.randint(0, 256, (300, 100, 3), dtype=np.uint8) # 100 by 300 noise

        # Map that image to physical space
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[-0.5,0.5], y_range_mm=[-1,1], pixel_size_mm=1e-3)
        y_size, x_size, _ = mapped_image.shape

        # Verify size is correct
        self.assertEqual(y_size,2*1/1e-3)
        self.assertEqual(x_size,2*0.5/1e-3)

        # Map that image to physical space (no mapping), see that pixel values are the same
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[0,3], pixel_size_mm=1e-2)
        self.assertAlmostEqual(random_image[0,0,0],mapped_image[0,0,0])
        self.assertAlmostEqual(random_image[10,10,1],mapped_image[10,10,1])
        self.assertAlmostEqual(random_image[50,50,2],mapped_image[50,50,2])

    def test_image_to_physical_translations_uv(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0]] # mm
        fp1 = FitPlaneElastic.from_points(uv,xyz)

        # Slightly different plane shifted over by 10
        uv = [[10,0],[110,0],[10,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0]] # mm
        fp2 = FitPlaneElastic.from_points(uv,xyz)

        np.random.seed(42)
        blank = np.zeros((300, 100, 3))
        blank[50,50] = np.array([255,255,255])
        blank[20,70] = np.array([255,0,0])

        fp1_image = fp1.image_to_physical(blank, [0,1], [0,1], 0.01)
        fp2_image = fp2.image_to_physical(blank, [0,1], [0,1], 0.01)

        # self.show_image(fp1_image)
        # self.show_image(fp2_image)

        # Find the position of the white pixel and make sure it is in the correct place
        npt.assert_array_almost_equal(fp1_image[50,50], [255,255,255])
        npt.assert_array_almost_equal(fp2_image[50,40], [255,255,255])

    def test_distance_metrics(self):

        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_center_positions_xyz_mm, print_inputs=False)

        # Define test inputs
        uv_pix = np.array([[0, 1], [1, 0]])  # maps to [[0, 1, 0], [1, 0, 0]]
        xyz_mm = np.array([[0, 2, 0.1], [1, 0, -0.1]])  

        in_plane, out_plane = fp.get_template_center_positions_distance_metrics(uv_pix, xyz_mm, mean=True)

        assert np.isclose(in_plane, 0.5)
        assert np.isclose(out_plane, 0.1)

        in_plane_indiv_pt, out_plane_indiv_pt = fp.get_template_center_positions_distance_metrics(uv_pix, xyz_mm, mean=False)
        npt.assert_array_almost_equal(in_plane_indiv_pt, np.array([1,0]))
        npt.assert_array_almost_equal(out_plane_indiv_pt, np.array([0.1, 0.1]))
