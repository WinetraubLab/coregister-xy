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
        self.target_centers_list = [[0,1000], [1000, 1000], [0, 0]]

    def test_main_function_runs(self):
        f = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.target_centers_list, template_size=401, um_per_pixel=2)

    def test_calc_fitplane_centers(self):
        tk_filepath = "plane/test_vectors/align3_int.xml"
        fp1 = ParseXML.from_imagej_xml(tk_filepath, 8, 11, 68, None, True)
        fp2 = ParseXML.from_imagej_xml(tk_filepath, 8, 14, 52, None, True)
        fp3 = ParseXML.from_imagej_xml(tk_filepath, 8, 17, 56, None, True)
        fp_list = [fp1, fp2, fp3]
        fmp = FitMultiPlane.from_aligned_fitplanes(fp_list, self.target_centers_list, template_size=0, um_per_pixel=4)
        centers  = [(project.tx + fmp.template_size/2, project.ty + fmp.template_size/2) for project in fmp.fitplanes]
        um_centers = fmp.calc_fitplane_centers()
        self.assertAlmostEqual(centers[1][0]*2, um_centers[1][0])
        self.assertAlmostEqual(centers[2][1]*2, um_centers[2][1])

    def test_print_stats(self):
        fmp = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.target_centers_list, template_size=401, um_per_pixel=2)
        fmp.get_single_template_stats()
       
    def test_fit_mapping(self):
        fmp = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.target_centers_list, template_size=401, um_per_pixel=2)
        # set known values
        fmp.fitplane_centers = np.array([[1,2,3], [2,2,3], [1,3,3]])
        fmp.target_centers = np.array([[2,2,3], [4,2,3], [2,3,3]])

        u,v,h = fmp.fit_mapping_to_xy()
        self.assertAlmostEqual(u[0], 2)
        self.assertAlmostEqual(u[1], 0)
        self.assertAlmostEqual(v[0], 0)
        self.assertAlmostEqual(v[1], 1)
        self.assertAlmostEqual(h[2], 1)

    def test_fit_mapping_2(self):
        fmp = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.target_centers_list, template_size=401, um_per_pixel=2)
        # set known values
        fmp.fitplane_centers = np.array([[0,0,3], [10,0,4], [0,10,5]])
        fmp.target_centers = np.array([[0,2,3], [10,2,4], [0,12,5]])

        u,v,h = fmp.fit_mapping_to_xy()
        
        a = fmp.get_xyz_from_uv([0,10,5])
        self.assertAlmostEqual(a[0], 0)
        self.assertAlmostEqual(a[1], 12)

        inplane_error = fmp.avg_in_plane_projection_error(fmp.fitplane_centers, fmp.target_centers)
        self.assertAlmostEqual(inplane_error, 0) # 3 points, so it should be an exact mapping
    
    def test_fit_mapping_approx(self):
        fmp = FitMultiPlane.from_aligned_fitplanes(self.fp_list, self.target_centers_list, template_size=401, um_per_pixel=2)
        # set known values
        fmp.fitplane_centers = np.array([[0,0,1], [10,0,6], [0,10,3], [5,5,5]])
        fmp.target_centers = np.array([[0,0,4], [10,0,4], [0,10,4], [5,5,4]])

        u,v,h = fmp.fit_mapping_to_xy()        
        a = fmp.get_xyz_from_uv([0,10,3])

        inplane_error = fmp.avg_in_plane_projection_error(fmp.fitplane_centers, fmp.target_centers)
        self.assertAlmostEqual(inplane_error, 0)      

        out_of_plane_error = fmp.avg_out_of_plane_projection_error(fmp.fitplane_centers, fmp.target_centers)
        e = np.mean([-3, 2, -1, 1])
        self.assertAlmostEqual(e, out_of_plane_error)

