import math
import numpy as np
import numpy.testing as npt
import unittest
from plane.fit_multi_plane import FitMultiPlane
from plane.parse_xml import ParseXML

class TestFitMultiPlane(unittest.TestCase):

    def setUp(self):
        tk_filepath = "plane/test_vectors/align3.xml"
        self.tk_filepath = tk_filepath
        fp1 = ParseXML.from_imagej_xml(tk_filepath, 8, 11, 68, None, True)
        fp2 = ParseXML.from_imagej_xml(tk_filepath, 8, 14, 52, None, True)
        fp3 = ParseXML.from_imagej_xml(tk_filepath, 8, 17, 56, None, True)

        self.fp_list = [fp1, fp2, fp3]
        self.real_centers_list = [[0,1000], [1000, 1000], [0, 0]]

    def test_main_function_runs(self):
        f = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.real_centers_list, template_size=401, um_per_pixel=2)

    def test_calc_fitplane_centers(self):
        tk_filepath = "plane/test_vectors/align3_int.xml"
        fp1 = ParseXML.from_imagej_xml(tk_filepath, 8, 11, 68, None, True)
        fp2 = ParseXML.from_imagej_xml(tk_filepath, 8, 14, 52, None, True)
        fp3 = ParseXML.from_imagej_xml(tk_filepath, 8, 17, 56, None, True)
        fp_list = [fp1, fp2, fp3]
        fmp = FitMultiPlane.from_aligned_fitplanes(fp_list, self.real_centers_list, template_size=0, um_per_pixel=4)
        centers  = [(project.tx + fmp.template_size/2, project.ty + fmp.template_size/2) for project in fmp.fitplanes]
        um_centers = fmp.calc_fitplane_centers()
        self.assertAlmostEqual(centers[1][0]*2, um_centers[1][0])
        self.assertAlmostEqual(centers[2][1]*2, um_centers[2][1])

    def test_print_stats(self):
        fmp = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.real_centers_list, template_size=401, um_per_pixel=2)
        fmp.print_single_plane_stats()
       
