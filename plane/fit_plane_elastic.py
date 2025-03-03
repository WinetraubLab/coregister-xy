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

        # Inverse mapping
        # Prevent singularity
        perturbed_template_positions_xyz_mm = template_positions_xyz_mm + np.random.normal(scale=1e-12, size=template_positions_xyz_mm.shape)
            
        xyz_to_uv_interpolator = RBFInterpolator(
            perturbed_template_positions_xyz_mm[:,:2],  # Use only x and y for inverse (2D)
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
    
    def get_uv_from_xyz(self, xyz_mm):
        """
        Map 3D xyz coordinates to 2D uv coordinates using the inverse interpolator.

        Args:
            xyz_mm: 3D xyz coordinates as a numpy array of shape (n, 3).

        Returns:
            2D uv coordinates as a numpy array of shape (n, 2).
        """
        xyz_mm = np.array(xyz_mm)
        if xyz_mm.ndim == 1:
            xyz_mm = xyz_mm[np.newaxis, :]  # Add batch dimension for single point
        return self.xyz_to_uv_interpolator(xyz_mm[:, :2])  # Use only x and y for inverse
    
    def image_to_physical(self, cv2_image, x_range_mm=[-1, 1], y_range_mm=[-1, 1], pixel_size_mm=1e-3):
        """
        Project a 2D image to 3D physical space within range x_range_mm, y_range_mm using TPS interpolation.

        Args:
            cv2_image: The source image (2D or 3D RGB) to be transformed.
            x_range_mm: The physical range in the x-direction (in mm).
            y_range_mm: The physical range in the y-direction (in mm).
            pixel_size_mm: The size of each pixel in mm.

        Returns:
            transformed_image: The transformed image in physical space.
        """
        # Input checks
        x_range_mm = np.array(x_range_mm)
        y_range_mm = np.array(y_range_mm)
        if x_range_mm[1] <= x_range_mm[0] or y_range_mm[1] <= y_range_mm[0]:
            raise ValueError("Invalid range: x_range_mm and y_range_mm must be increasing")
        if pixel_size_mm <= 0:
            raise ValueError("pixel_size_mm must be positive")

        # Calculate image dimensions
        width_px = int((x_range_mm[1] - x_range_mm[0]) / pixel_size_mm)
        height_px = int((y_range_mm[1] - y_range_mm[0]) / pixel_size_mm)

        # Define the destination grid in physical coordinates
        x_mm = np.linspace(x_range_mm[0], (x_range_mm[1]/pixel_size_mm - 1) * pixel_size_mm, width_px)
        y_mm = np.linspace(y_range_mm[0], (y_range_mm[1]/pixel_size_mm - 1) * pixel_size_mm, height_px)
        xx_mm, yy_mm = np.meshgrid(x_mm, y_mm)

        # Flatten the grid for TPS transformation
        mm_points = np.vstack([xx_mm.ravel(), yy_mm.ravel()]).T

        # Map physical coordinates to UV coordinates using the inverse interpolator
        uv_points = self.get_uv_from_xyz(mm_points)

        # Reshape UV coordinates to match the destination grid
        uv_points = uv_points.reshape((height_px, width_px, 2))

        # Extract U and V coordinates
        u_coords = uv_points[:, :, 0]
        v_coords = uv_points[:, :, 1]

        # RGB images: Apply map_coordinates to each channel separately
        if len(cv2_image.shape) == 3:
            warped_channels = [
                map_coordinates(
                    cv2_image[:, :, channel],  # Extract one channel
                    [v_coords, u_coords],     # Use UV coordinates
                    order=1,                  # Bilinear interpolation
                    mode='constant',          # Fill with zeros outside boundaries
                    cval=0.0                 # Fill value
                )
                for channel in range(cv2_image.shape[2])  # Loop over channels
            ]
            # Stack the warped channels back into a 3D image
            transformed_image = np.stack(warped_channels, axis=-1)
        else:  # Grayscale
            transformed_image = map_coordinates(
                cv2_image,
                [v_coords, u_coords],
                order=1,
                mode='constant',
                cval=0.0
            )

        return transformed_image
    
    def get_xyz_points_positions_distance_metrics(self, uv_pix, xyz_mm, mean=True):
        """ 
        uv_pix: coordinates in pixels, array shape (2,n)
        xyz_mm: coordinates in mm, array shape (3,n)
        mean: if True, average over all points. If False, return individual error per point
        Returns in plane and out of plane distances between mapped uv points and corresponding xyz points.
        """
        # Input check
        uv_pix = np.array(uv_pix)
        xyz_mm = np.array(xyz_mm)
        assert uv_pix.shape[0] == xyz_mm.shape[0]

        uv_to_xyz = np.squeeze(np.array([self.get_xyz_from_uv(p) for p in uv_pix]))
        in_plane = np.sqrt(np.sum((uv_to_xyz[:,:2] - xyz_mm[:,:2])**2, axis=1))
        out_plane = np.abs(uv_to_xyz[:, 2] - xyz_mm[:, 2]) # Avg differences on z
        if mean:
            return np.mean(in_plane), np.mean(out_plane)
        else:
            return in_plane, out_plane
    