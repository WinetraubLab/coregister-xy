import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane_elastic import FitPlaneElastic
import cv2

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.fluorescent_image_points_positions_uv_pix = [[0, 1], [1, 0], [1, 1], [0.5, 0.5]]
        self.template_positions_xyz_mm = [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 0]]

    def test_main_function_runs(self):
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=True)
    
    def show_image(self, image): # Use for debugging
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_get_xyz_from_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        # Check that transformation projects uv to xyz 
        npt.assert_array_almost_equal(xyz, self.template_positions_xyz_mm)

    def test_get_uv_from_xyz(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        uv = fp.get_uv_from_xyz(self.template_positions_xyz_mm)
        # Check that transformation projects xyz to uv
        npt.assert_array_almost_equal(uv, self.fluorescent_image_points_positions_uv_pix)

    def test_distance_from_elastic_to_affine(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        e_in_plane, e_out_plane = fp.get_elastic_affine_diff_mm(self.fluorescent_image_points_positions_uv_pix)

        zeros = np.tile([0, 0, 0], (len(self.fluorescent_image_points_positions_uv_pix), 1))
        npt.assert_array_almost_equal(e_in_plane, zeros, decimal=3) # Almost equal to 1um
        npt.assert_array_almost_equal(e_out_plane, zeros, decimal=3)

    def test_uv_to_xyz_back_to_uv_with_smoothing(self):
        rand = np.random.rand(np.array(self.template_positions_xyz_mm).shape[0], np.array(self.template_positions_xyz_mm).shape[1])
        template_positions_xyz_mm_perturbed = self.template_positions_xyz_mm + rand
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, template_positions_xyz_mm_perturbed, smoothing=1e-4, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        uv = fp.get_uv_from_xyz(xyz)
        npt.assert_array_almost_equal(uv, self.fluorescent_image_points_positions_uv_pix, decimal=3)

    def test_uv_to_xyz_back_to_uv(self):
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm, print_inputs=False)
        xyz = fp.get_xyz_from_uv(self.fluorescent_image_points_positions_uv_pix)
        uv = fp.get_uv_from_xyz(xyz)
        npt.assert_array_almost_equal(uv, self.fluorescent_image_points_positions_uv_pix)

    def test_split_vector_to_in_plane_and_out_plane_physical_coordinates(self):
        # Create a plane that is parallel to xy
        uv = [[0, 0], [100, 0], [0, 300], [100, 300]]  # pix
        xyz = [[0, 0, 0], [1, 0, 0], [0, 3, 0], [1, 3, 0]]  # mm
        fp = FitPlaneElastic.from_points(uv, xyz)

        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane([1,2,3],output_coordinate_system='physical')

        # Check dimensions (one vector)
        self.assertAlmostEqual(in_p.shape[0],3)
        self.assertAlmostEqual(len(in_p.shape), 1)
        self.assertAlmostEqual(out_p.shape[0], 3)
        self.assertAlmostEqual(len(out_p.shape), 1)

        # Check component split
        self.assertAlmostEqual(in_p[0], 1)
        self.assertAlmostEqual(in_p[1], 2)
        self.assertAlmostEqual(in_p[2], 0)
        self.assertAlmostEqual(out_p[0], 0)
        self.assertAlmostEqual(out_p[1], 0)
        self.assertAlmostEqual(out_p[2], 3)

        # Check sign
        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane([1, 2, -3])
        self.assertAlmostEqual(out_p[2], -3)
        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane([-1, 2, 3])
        self.assertAlmostEqual(in_p[0], -1)

        # Check vector operation
        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane([[1, 2, 3],[4, 5, 6]])
        self.assertAlmostEqual(in_p.shape[0], 2)
        self.assertAlmostEqual(in_p.shape[1], 3)
        self.assertAlmostEqual(out_p.shape[0], 2)
        self.assertAlmostEqual(out_p.shape[1], 3)
        self.assertAlmostEqual(in_p[0, 0], 1)
        self.assertAlmostEqual(in_p[0, 1], 2)
        self.assertAlmostEqual(in_p[1, 0], 4)
        self.assertAlmostEqual(in_p[1, 1], 5)
        self.assertAlmostEqual(out_p[0, 2], 3)
        self.assertAlmostEqual(out_p[1, 2], 6)

    def test_split_vector_to_in_plane_and_out_plane_plane_coordinates(self):
        # Create a plane that is parallel to xy
        uv = [[0, 0], [100, 0], [0, 300], [100, 300]]  # pix
        xyz = [[0, 0, 0], [1, 0, 0], [0, 3, 0], [1, 3, 0]]  # mm
        fp = FitPlaneElastic.from_points(uv, xyz)

        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane(
            [[1, 2, 3],[4, 5, 6],[-1,-2,-3]],output_coordinate_system='plane')

        # Check sizes
        self.assertAlmostEqual(in_p.shape[0], 3)
        self.assertAlmostEqual(in_p.shape[1], 2)
        self.assertAlmostEqual(out_p.shape[0], 3)
        self.assertAlmostEqual(len(out_p.shape), 1)

        # Check projection values
        self.assertAlmostEqual(in_p[0,0], 1)
        self.assertAlmostEqual(in_p[2, 0], -1)
        self.assertAlmostEqual(in_p[1, 1], 5)
        self.assertAlmostEqual(out_p[0], 3)
        self.assertAlmostEqual(out_p[2], -3)

        # Force normal to y axis and make sure that values are maintained
        in_p, out_p = fp._split_vector_to_in_plane_and_out_plane(
            [[1, 2, 3], [-1, -2, -3]], forced_plane_normal=[0,-1,0], output_coordinate_system='plane')
        self.assertAlmostEqual(in_p[0, 0], 1)
        self.assertAlmostEqual(in_p[0, 1], 3)
        self.assertAlmostEqual(in_p[1, 1], -3)
        self.assertAlmostEqual(out_p[0], -2)
        self.assertAlmostEqual(out_p[1], 2)

    def test_image_to_physical_translations_xy(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300], [100,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0], [1,3,0]] # mm
        fp = FitPlaneElastic.from_points(uv,xyz)
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
        
        # self.show_image(random_image)
        # self.show_image(mapped_image)
        
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

    def test_transform_grid(self):
        """Create a test case warping points to an integer-coordinate grid.
        TPS is ill-conditioned for transformations when grid is regular. This test case
        checks whether it fails under these conditions. """

        uv_pix = [[96.0, 61.0], [236.833, 146.167], [379.5, 203.5], [511.5, 243.5], [628.833, 270.167], [731.5, 288.833], [822.167, 299.5], [898.167, 310.167], [962.167, 315.5], [1012.833, 319.5], [95.5, 239.5], [238.167, 283.5], [376.833, 312.833], [508.833, 331.5], [628.833, 343.5], [731.5, 352.833], [820.833, 359.5], [898.167, 364.833], [962.167, 368.833], [1014.167, 370.167], [95.5, 419.5], [236.833, 419.5], [376.833, 419.5], [511.5, 419.5], [627.5, 419.5], [732.833, 419.5], [822.167, 418.167], [895.5, 418.167], [962.167, 419.5], [1015.5, 419.5], [95.5, 598.167], [238.167, 556.833], [379.5, 526.167], [510.167, 507.5], [630.167, 494.167], [732.833, 484.833], [819.5, 479.5], [898.167, 474.167], [960.833, 470.167], [1014.167, 468.833]]
        xyz_mm = [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0], [5.0, 1.0, 0.0], [6.0, 1.0, 0.0], [7.0, 1.0, 0.0], [8.0, 1.0, 0.0], [9.0, 1.0, 0.0], [10.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [4.0, 2.0, 0.0], [5.0, 2.0, 0.0], [6.0, 2.0, 0.0], [7.0, 2.0, 0.0], [8.0, 2.0, 0.0], [9.0, 2.0, 0.0], [10.0, 2.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0], [4.0, 3.0, 0.0], [5.0, 3.0, 0.0], [6.0, 3.0, 0.0], [7.0, 3.0, 0.0], [8.0, 3.0, 0.0], [9.0, 3.0, 0.0], [10.0, 4.0, 0.0], [1.0, 4.0, 0.0], [2.0, 4.0, 0.0], [3.0, 4.0, 0.0], [4.0, 4.0, 0.0], [5.0, 4.0, 0.0], [6.0, 4.0, 0.0], [7.0, 4.0, 0.0], [8.0, 4.0, 0.0], [9.0, 4.0, 0.0], [10.0, 4.0, 0.0]]

        with self.assertRaises(AssertionError) as context:
            FitPlaneElastic.from_points(uv_pix, xyz_mm)
        self.assertIn("Inverse consistency check failed", str(context.exception))

    def test_get_normal(self):
        xyz_mm = np.array([
            [0, 0, 0],  
            [1, 0, 1],   
            [0, 1, 1],   
            [1, 1, 2],   
            [0.5, 0.5, 1.5]  
        ])
        fp = FitPlaneElastic.from_points(self.fluorescent_image_points_positions_uv_pix, self.template_positions_xyz_mm)
        npt.assert_almost_equal([0,0,1], fp.normal())

        # tilted example
        xyz_mm = np.array([
            [0, 0, 0],  
            [1, 0, 1],   
            [0, 1, 1],   
            [1, 1, 2],   
            [0.5, 0.5, 1.5]  
        ])
        fp = FitPlaneElastic.from_points(np.random.rand(xyz_mm.shape[0], 2), xyz_mm)
        npt.assert_almost_equal([-0.58961147, -0.58961147,  0.55201144], fp.normal())

    def test_plots(self):
        uv = np.array([[0, 0], [100, 0], [200, 200], [0, 300]])  # pix
        xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 2, 0], [0, 3, 0]])  # mm

        # add some noise to xyz
        np.random.seed(42)
        xyz = xyz + np.random.normal(0,0.25,xyz.shape)

        fp = FitPlaneElastic.from_points(uv, xyz)
        fp.plot_explore_anchor_points_fit_quality('With Elastic Fit, should be close', use_elastic_fit=True)
        fp.plot_explore_anchor_points_fit_quality('Only Affine, should be further', use_elastic_fit=False)
        fp.plot_explore_anchor_points_fit_quality('Only Affine, In Plane', use_elastic_fit=False, coordinate_system='plane')
