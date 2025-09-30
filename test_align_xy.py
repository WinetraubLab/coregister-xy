import numpy as np
import numpy.testing as npt
import unittest
import cv2

import cell_to_cell_registration.align_xy as align_xy

class TestAlignXY(unittest.TestCase):
    def test_rough_alignment(self):
        histology_coords_px = np.array([
            [75,50],
            [75,2],
            [25,50],
            [25,2]
        ])
        oct_coords_mm = np.array([
            [0.05, -0.25],
            [0.15, -0.25],
            [0.05, -0.35],
            [0.15, -0.35],
        ])
        oct_crop_region_mm = np.array([
            [0.025, -0.125],
            [0.175, -0.375],
            [0.175, -0.125],
            [0.025, -0.375],
        ])
        histology_image = cv2.imread('example_data/test_histology_image.jpg')
        histology_image = cv2.cvtColor(histology_image, cv2.COLOR_BGR2RGB)

        w = align_xy.align_and_crop_histology_image(histology_image, oct_coords_mm, histology_coords_px, 
                                   oct_crop_region_mm, align_mode='affine')
        npt.assert_allclose(w.shape, (250,150,3))
        npt.assert_allclose(w[100,50], [255,255,255])
        npt.assert_allclose(w[45,100], [0,0,255], atol=15) # for low quality jpeg noise
