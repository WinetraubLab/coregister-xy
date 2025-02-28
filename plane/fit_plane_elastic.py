import numpy as np
import json
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from sklearn.metrics import mean_absolute_error

class FitPlaneElastic:
    """
    A class to perform 2D-to-3D Thin Plate Spline (TPS) transformations using RBFInterpolator.
    Supports forward mapping (uv -> xyz) and reverse mapping (xyz -> uv) using separate interpolators.
    """
    
    def __init__(self, uv_to_xyz_interpolator=None, xyz_to_uv_interpolator=None, control_points_uv_pix=None):
        """
        Initialize the FitPlaneElastic class.

        Args:
            uv_to_xyz_interpolator: An RBFInterpolator object for forward mapping (uv -> xyz).
            xyz_to_uv_interpolator: An RBFInterpolator object for reverse mapping (xyz -> uv).
            control_points_uv_pix: The control points used to initialize the interpolators.
        """
        self.uv_to_xyz_interpolator = uv_to_xyz_interpolator  # Forward interpolator (uv -> xyz)
        self.xyz_to_uv_interpolator = xyz_to_uv_interpolator  # Inverse interpolator (xyz -> uv)
        self.control_points_uv_pix = control_points_uv_pix  # Control points (uv and xyz)
        if self.control_points_uv_pix is not None:
            self.control_points_uv_pix = np.array(self.control_points_uv_pix)
    
    @classmethod
    def from_points(cls, fluorescent_image_points_uv_pix, template_positions_xyz_mm, print_inputs=False):
        """
        Initialize a FitPlaneElastic object using control points.

        Args:
            fluorescent_image_points_uv_pix: 2D source points (uv) as a numpy array of shape (n, 2).
            template_positions_xyz_mm: 3D target points (xyz) as a numpy array of shape (n, 3).
            print_inputs: If True, print the inputs for debugging.

        Returns:
            A FitPlaneElastic object.
        """
        # Input validation
        fluorescent_image_points_uv_pix = np.array(fluorescent_image_points_uv_pix)
        template_positions_xyz_mm = np.array(template_positions_xyz_mm)
        if fluorescent_image_points_uv_pix.shape[0] != template_positions_xyz_mm.shape[0]:
            raise ValueError("Number of points should be the same between fluorescent_image_points_uv_pix and template_positions_xyz_mm")
        if fluorescent_image_points_uv_pix.shape[1] != 2:
            raise ValueError("fluorescent_image_points_uv_pix must have shape (n, 2)")
        if template_positions_xyz_mm.shape[1] != 3:
            raise ValueError("template_positions_xyz_mm must have shape (n, 3)")

        # Print inputs for debugging
        if print_inputs:
            print("fluorescent_image_points_uv_pix:\n", fluorescent_image_points_uv_pix)
            print("template_positions_xyz_mm:\n", template_positions_xyz_mm)

        # Create forward interpolator (uv -> xyz)
        uv_to_xyz_interpolator = RBFInterpolator(
            fluorescent_image_points_uv_pix,  # 2D source points (uv)
            template_positions_xyz_mm,  # 3D target points (xyz)
            kernel='thin_plate_spline',  
            neighbors=None  # Use all points for interpolation
        )

        # Create inverse interpolator (xyz -> uv)
        xyz_to_uv_interpolator = RBFInterpolator(
            template_positions_xyz_mm[:, :2],  # Use only x and y for inverse (2D)
            fluorescent_image_points_uv_pix,  
            kernel='thin_plate_spline', 
            neighbors=None
        )

        # Store control points
        control_points_uv_pix = fluorescent_image_points_uv_pix

        return cls(uv_to_xyz_interpolator, xyz_to_uv_interpolator, control_points_uv_pix)
    
    def get_xyz_from_uv(self, uv_pix):
        """
        Map 2D uv coordinates to 3D xyz coordinates using the forward interpolator.

        Args:
            uv_pix: 2D uv coordinates as a numpy array of shape (n, 2).

        Returns:
            3D xyz coordinates as a numpy array of shape (n, 3).
        """
        uv_pix = np.array(uv_pix)
        if uv_pix.ndim == 1:
            uv_pix = uv_pix[np.newaxis, :]  # Add batch dimension for single point
        return self.uv_to_xyz_interpolator(uv_pix)
    