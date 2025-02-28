import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

class FitPlaneElastic:
    
    """ Begin constractor methods """
    def __init__(self, tps_interpolator=None, control_points_uv_pix=None):
        self.tps_interpolator = tps_interpolator
        self.control_points_uv_pix = control_points_uv_pix
        if not (self.control_points_uv_pix is None):
            self.control_points_uv_pix = np.array(self.control_points_uv_pix)
    
    @classmethod
    def from_points(
        cls, fluorescent_image_points_uv_pix, template_positions_xyz_mm, print_inputs = False):
        """
        This function initializes a FitPlaneElastic by applying Thin Plate Spline on a list of points.

        INPUTS:
            fluorescent_image_points_uv_pix: For each photobleach barcode, find the center position in pixels. This is an
                array of these center points [[x1, y1], [x2, y2],..., [xn, yn]] with shape (n,2)
            template_positions_xyz_mm: An array [[x1, y1, z1],..., [xn, yn, zn]] of shape (n,3) containing points defining the 
                position (in um) of the locations that each of the points in fluorescent_image_points_uv_pix should map to. 
                These points can be obtained from the photobleaching script.
            print_inputs: prints to screen the inputs of the function for debug purposes.
        """
        # Input check
        fluorescent_image_points_uv_pix = np.array(fluorescent_image_points_uv_pix)
        template_positions_xyz_mm = np.array(template_positions_xyz_mm)
        if (fluorescent_image_points_uv_pix.shape[0] != template_positions_xyz_mm.shape[0]):
            raise ValueError("Number of points should be the same between " + 
                "fluorescent_image_points_uv_pix, template_positions_xyz_mm")
        if fluorescent_image_points_uv_pix.shape[1] != 2:
            raise ValueError("Number of elements in fluorescent_image_points_uv_pix should be two")
        if template_positions_xyz_mm.shape[1] != 3:
            raise ValueError("Number of elements in template_positions_xyz_mm should be three")
        
        # Print inputs
        if print_inputs:
            txt = ("FitPlane.from_template_centers(" +
                   json.dumps(fluorescent_image_points_uv_pix.tolist()) + "," +
                   json.dumps(template_positions_xyz_mm.tolist()))   
            txt += ')'
            print(txt)

        # TPS interpolators for each x, y, z 
        tps_interpolator = RBFInterpolator(
            fluorescent_image_points_uv_pix,  # 2D input points (u, v)
            template_positions_xyz_mm,  # 3D target points (x, y, z)
            kernel='thin_plate_spline',  # Use Thin Plate Spline kernel
            neighbors=None  # Use all points for interpolation
        )
        control_points_uv_pix = fluorescent_image_points_uv_pix

        return cls(tps_interpolator, control_points_uv_pix)
    