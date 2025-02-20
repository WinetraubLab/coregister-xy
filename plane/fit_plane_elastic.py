import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import cv2
from scipy.interpolate import Rbf, griddata

class FitPlaneElastic:
    
    """ Begin constractor methods """
    def __init__(self, tps_weights=None, control_points=None):
        self.tps_weights = tps_weights
        self.control_points = control_points
        if not (self.tps_weights is None or self.control_points is None):
            self.tps_weights = np.array(self.tps_weights)
            self.control_points = np.array(self.control_points)
    
    @classmethod
    def from_points(
        cls, template_positions_uv_pix, template_positions_xyz_mm, print_inputs = False):
        """
        This function initializes a FitPlaneElastic by applying Thin Plate Spline on a list of points.

        INPUTS:
            template_positions_uv_pix: For each photobleach barcode, find the center position in pixels. This is an
                array of these center points [[x1, y1], [x2, y2],..., [xn, yn]] with shape (n,2)
            template_positions_xyz_mm: An array [[x1, y1, z1],..., [xn, yn, zn]] of shape (n,3) containing points defining the 
                position (in um) of the locations that each of the points in template_positions_uv_pix should map to. 
                These points can be obtained from the photobleaching script.
            print_inputs: prints to screen the inputs of the function for debug purposes.
        """
        # Input check
        template_positions_uv_pix = np.array(template_positions_uv_pix)
        template_positions_xyz_mm = np.array(template_positions_xyz_mm)
        if (template_positions_uv_pix.shape[0] != template_positions_xyz_mm.shape[0]):
            raise ValueError("Number of points should be the same between " + 
                "template_positions_uv_pix, template_positions_xyz_mm")
        if template_positions_uv_pix.shape[1] != 2:
            raise ValueError("Number of elements in template_positions_uv_pix should be two")
        if template_positions_xyz_mm.shape[1] != 3:
            raise ValueError("Number of elements in template_positions_xyz_mm should be three")
        
        # Print inputs
        if print_inputs:
            txt = ("FitPlane.from_template_centers(" +
                   json.dumps(template_positions_uv_pix.tolist()) + "," +
                   json.dumps(template_positions_xyz_mm.tolist()))   
            txt += ')'
            print(txt)

        # Compute Thin Plate Spline transformation
        x = template_positions_xyz_mm[:, 0]
        y = template_positions_xyz_mm[:, 1]
        z = template_positions_xyz_mm[:, 2]

        # TPS interpolators for each x, y, z 
        tps_x = Rbf(template_positions_uv_pix[:, 0], template_positions_uv_pix[:, 1], x, function='thin_plate')
        tps_y = Rbf(template_positions_uv_pix[:, 0], template_positions_uv_pix[:, 1], y, function='thin_plate')
        tps_z = Rbf(template_positions_uv_pix[:, 0], template_positions_uv_pix[:, 1], z, function='thin_plate')

        # Store weights and control points
        tps_weights = np.vstack([tps_x.nodes, tps_y.nodes, tps_z.nodes])
        control_points = template_positions_uv_pix

        return cls(tps_weights, control_points)