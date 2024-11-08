import numpy as np
# from plane.fit_plane import FitPlane
from plane.fit_template import FitTemplate
import pandas as pd

class FitPlane:
    def __init__(self, fitplanes, target_centers, template_size, um_per_pixel):
        self.fitplanes = fitplanes
        self.real_centers = target_centers # in um
        self.template_size = template_size
        self.um_per_pixel = um_per_pixel
        self.fitplane_centers = self.calc_fitplane_centers() # in um
        self.distances = self.calc_distances() # in um

    @classmethod
    def from_aligned_fit_templates(cls, fit_templates_list, target_centers_list, template_size=401, um_per_pixel=2):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 

        :param fit_templates_list: list of barcodes contained in this FitPlane. Each is a FitTemplate object.
        :param target_centers_list: theoretical positions of each barcode center as defined by photobleach script.
        :param template_size: square edge length of the template image used for alignment in each FitTemplate, in pixels.
        :param um_per_pixel: um per pixel in the template image.
        :returns: Initializes an instance of a FitPlane.
        """
        return cls(fit_templates_list, target_centers_list, template_size, um_per_pixel)

    def __len__(self):
        return len(self.fitplanes)
    
    def calc_distances(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        Units are in um.
        """
        pass

    def calc_fitplane_centers(self):
        """
        Convert Fitplane centers in pixels to distance in um, and add z coordinate for each.
        """
        pass

    def fit_from_photobleach(self):
        """
        Calculate a mapping to project pixels from the angled tissue slice onto a flat plane (match with the photobleach template).
        UVH mapping: for a point (u,v,z') on the sliced tissue, [x,y,z] = vec_u * u + vec_v * v + vec_h
        Prints and returns vectors U, V, and H.
        """
        pass

    def print_single_plane_stats(self):
        """
        Prints stats for each FitPlane as a table: shrinkage, rotation, shear, and mean/stdev for each
        Units: um
        """
        pass

    def project_centers_onto_flat_plane(self):
        """
        Project each (u, v, z') pair from the barcode centers on an angled tissue slice onto a flat xy plane using the 
        vectors UVH from fit_mapping_to_xy. Z for the flat plane is 0.
        Returns: array-like of transformed points.
        """
        pass
    
    def compute_avg_projection_error(a, b):
        """
        Returns the mean distance between points in arrays a and b, for evaluating best-fit calculated projection.
        Ignores z coordinate.
        """
        pass
