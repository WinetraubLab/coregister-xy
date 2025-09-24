import numpy as np
import numpy.testing as npt
import unittest
import cv2
import tifffile

import segment_cells

class TestPCR99a(unittest.TestCase):

    def test_cellseg2d(self):
        img_2d = cv2.imread('example_data/2d-sample-image.jpg')
        masks, flows = segment_cells(img_2d, 10, dim=2, flow_threshold=0.7, cellprob_threshold=-5, keep_dark_cells=True, gpu=True, normalization="clahe")
        assert masks.max() > 15

    def test_cellseg3d(self):
        img_3d = tifffile.imread('example_data/3d-sample-image.tiff')
        masks, flows = segment_cells(img_3d, 10, dim=3, flow_threshold=0.7, cellprob_threshold=-5, keep_dark_cells=True, gpu=True, normalization="global")
        # acceptable range
        assert masks.max() > 15
        assert masks.max() < 35

    def test_filter_dark(self):
        img_2d = cv2.imread('example_data/2d-sample-image.jpg')
        unfiltered_masks, _ = segment_cells(img_2d, 10, dim=2, flow_threshold=0.7, cellprob_threshold=-5, keep_dark_cells=False, gpu=True, normalization="global")
        filtered_masks, _ = segment_cells(img_2d, 10, dim=2, flow_threshold=0.7, cellprob_threshold=-5, keep_dark_cells=True, gpu=True, normalization="global")
        # There should be 2 masks filtered out
        assert unfiltered_masks.max()-filtered_masks.max() == 2
