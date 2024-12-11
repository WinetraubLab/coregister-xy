
import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_plane import FitPlane
import cv2

class TestFitPlane(unittest.TestCase):

    def setUp(self):
        self.template_center_positions_uv_pix = [[0,1],[1,0],[1,1]]
        self.template_center_positions_xyz_um = [[0,1,0],[1,0,0],[1,1,0]]

    def test_main_function_runs(self):
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um, print_inputs=False)
        FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um, print_inputs=True)

    def test_fit_mapping(self):
        fp = FitPlane.from_template_centers(self.template_center_positions_uv_pix, self.template_center_positions_xyz_um, print_inputs=False)
        self.assertAlmostEqual(fp.u[0], 1)
        self.assertAlmostEqual(fp.u[1], 0)
        self.assertAlmostEqual(fp.u[2], 0)
        self.assertAlmostEqual(fp.v[0], 0)
        self.assertAlmostEqual(fp.v[1], 1)
        self.assertAlmostEqual(fp.v[2], 0)
        self.assertAlmostEqual(fp.h[0], 0)
        self.assertAlmostEqual(fp.u[1], 0)
        self.assertAlmostEqual(fp.u[2], 0)

        # Construct a case with randomly specified u,v,h vectors
        u = [2,1,0]
        v = [0,3,2]
        h = [10,15,12]
        xyz = []
        for p in self.template_center_positions_uv_pix:
            p2 = np.array(u) * p[0] + np.array(v) * p[1] + np.array(h)
            xyz.append(p2)
        
        fp = FitPlane.from_template_centers(self.template_center_positions_uv_pix, xyz, print_inputs=False)
        for i in range(0,3):
                self.assertAlmostEqual(u[i], fp.u[i])
                self.assertAlmostEqual(v[i], fp.v[i])
                self.assertAlmostEqual(h[i], fp.h[i])


    def test_error_raised_when_input_shape_is_wrong(self):
        # Check number of elements in uv vector different from number of elements in xyz vector 
        with self.assertRaises(ValueError):    
            FitPlane.from_template_centers([[0,1],[0,2],[0,3]],[[0,1,0],[0,2,0]])

        # Check number of elements in uv vector is not two
        with self.assertRaises(ValueError):
            FitPlane.from_template_centers([[0,1],[0,2],[0,3]],[[0,1],[0,2],[0,3]])

        # Check number of elements in xyz vector is not three
        with self.assertRaises(ValueError):
            FitPlane.from_template_centers([[0,1,0],[0,2,0],[0,3,0]],[[0,1,0],[0,2,0],[0,3,0]])

    def test_fit_with_force_normal_constrains(self):
        uv = [[0,0],[1,0],[0,1]]
        xyz = [[0,0,0],[1,0,0],[0,1,0]]

        n = np.array([0,0.5,1])
        n = n / np.linalg.norm(n)

        # Make sure un-forced version doesn't point in the norm direction.
        # This is an evaluation of the test's effectiveness.
        fp = FitPlane.from_template_centers(uv,xyz)
        self.assertLess(np.dot(n,np.array([0,0,1])),0.9)

        # Make sure plane fitted norm fits the desired direction.
        fp_n = FitPlane.from_template_centers(uv,xyz, forced_plane_normal = n)
        self.assertAlmostEqual(np.dot(fp_n.normal_direction(),n),1, places=1)

        # Fit with constrain, but the constrain is not needed, as it is being satisfied anyways.
        fp2 = FitPlane.from_template_centers(uv,xyz, forced_plane_normal = fp.normal_direction())
        self.assertAlmostEqual(fp.u[0],fp2.u[0])
        self.assertAlmostEqual(fp.u[1],fp2.u[1])
        self.assertAlmostEqual(fp.u[2],fp2.u[2])
        self.assertAlmostEqual(fp.v[0],fp2.v[0])
        self.assertAlmostEqual(fp.v[1],fp2.v[1])
        self.assertAlmostEqual(fp.v[2],fp2.v[2])
        self.assertAlmostEqual(fp.h[0],fp2.h[0])
        self.assertAlmostEqual(fp.h[1],fp2.h[1])
        self.assertAlmostEqual(fp.h[2],fp2.h[2])

    def test_image_to_physical_output_size(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0]] # mm
        fp = FitPlane.from_template_centers(uv,xyz)
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

    def test_image_to_physical_translations(self):
        # Create dummy plane with a random image
        uv = [[0,0],[100,0],[0,300]] # pix
        xyz = [[0,0,0],[1,0,0],[0,3,0]] # mm
        fp = FitPlane.from_template_centers(uv,xyz)
        np.random.seed(42)
        random_image = np.random.randint(0, 256, (300, 100, 3), dtype=np.uint8) # 100 by 300 noise

        # Map that image to the right, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[-0.1,0.9], y_range_mm=[0,3], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],0)
        self.assertAlmostEqual(mapped_image[0,0,1],0)
        self.assertAlmostEqual(mapped_image[0,0,2],0)
        self.assertAlmostEqual(mapped_image[11,0,0],0)
        self.assertGreater(mapped_image[10,30,0],0) # Area which shouldn't be effected

        # Map that image to the left, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0.1,1.1], y_range_mm=[0,3], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,100-5,0],0)
        self.assertAlmostEqual(mapped_image[0,100-5,1],0)
        self.assertAlmostEqual(mapped_image[0,100-5,2],0)
        self.assertAlmostEqual(mapped_image[10,100-5,0],0)
        self.assertGreater(mapped_image[10,100-11,0],0) # Area which shouldn't be effected

        # Map that image down, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[-0.1,2.9], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],0)
        self.assertAlmostEqual(mapped_image[0,0,1],0)
        self.assertAlmostEqual(mapped_image[0,0,2],0)
        self.assertAlmostEqual(mapped_image[0,11,0],0)
        self.assertGreater(mapped_image[30,11,0],0) # Area which shouldn't be effected

        # Map that image up, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[300-5,0,0],0)
        self.assertAlmostEqual(mapped_image[300-5,0,1],0)
        self.assertAlmostEqual(mapped_image[300-5,0,2],0)
        self.assertAlmostEqual(mapped_image[300-5,10,0],0)
        self.assertGreater(mapped_image[300-11,10,0],0) # Area which shouldn't be effected

        # Map that image up, see that filled with black
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0,1], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[300-5,0,0],0)
        self.assertAlmostEqual(mapped_image[300-5,0,1],0)
        self.assertAlmostEqual(mapped_image[300-5,0,2],0)
        self.assertAlmostEqual(mapped_image[300-5,10,0],0)
        self.assertGreater(mapped_image[300-11,10,0],0) # Area which shouldn't be effected

        # Shift to arbitrary point
        mapped_image = fp.image_to_physical(random_image, x_range_mm=[0.2,1.2], y_range_mm=[0.1,3.1], pixel_size_mm=1e-2)
        #self.show_image(mapped_image) # uncomment for debug
        self.assertAlmostEqual(mapped_image[0,0,0],random_image[10,20,0])
        

    def show_image(self, image): # Use for debugging
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
