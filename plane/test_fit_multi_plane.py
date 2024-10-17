import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_multi_plane import FitMultiPlane
from plane.parse_xml import ParseXML

class TestFitMultiPlane(unittest.TestCase):

    def setUp(self):
        self.tk_filepath = "plane/test_vectors/align-1.xml"

    def test_main_runs(self):
        landmarks = []
        real_centers = []
        FitMultiPlane.from_aligned_landmarks(landmarks, real_centers)

    def test_calc_dists(self):
        fp1 = ParseXML.from_imagej_xml(self.tk_filepath, 8, 11, None)
        fp2 = ParseXML.from_imagej_xml(self.tk_filepath, 8, 11, None)
        fp3 = ParseXML.from_imagej_xml(self.tk_filepath, 8, 11, None)

        fp3.tx = 10
        fp3.ty = 10
        fp2.tx = 3
        fp2.ty = 4
        fp1.tx = 0
        fp1.ty = 0

        landmarks = [fp1, fp2, fp3]
        real_centers = [(0, 50), (50, 100), (0,0)]
        mp = FitMultiPlane.from_aligned_landmarks(landmarks, real_centers)

        adj = mp.calc_distances()
        assert adj[1,0] == adj[0,1]
        self.assertAlmostEqual(adj[1,0], 5)