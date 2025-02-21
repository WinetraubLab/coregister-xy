import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

class FitPlaneElastic:
    
    """ Begin constractor methods """
    def __init__(self, tps_interpolator=None, control_points=None):
        self.tps_interpolator = tps_interpolator
        self.control_points = control_points
        if not (self.control_points is None):
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
        tps_interpolator = RBFInterpolator(
            template_positions_uv_pix,  # 2D input points (u, v)
            template_positions_xyz_mm,  # 3D target points (x, y, z)
            kernel='thin_plate_spline',  # Use Thin Plate Spline kernel
            neighbors=None  # Use all points for interpolation
        )
        control_points = template_positions_uv_pix

        return cls(tps_interpolator, control_points)
    
    def get_xyz_from_uv(self, uv_pix):
        """ Map a 2D point (u,v) to 3D (x,y,z) space using TPS interpolation. """
        uv_pix = np.array(uv_pix)
        xyz = self.tps_interpolator(uv_pix)  # Add batch dimension
        return xyz
  
    def image_to_physical(self, cv2_image, x_range_mm=[-1, 1], y_range_mm=[-1, 1], pixel_size_mm=1e-3):
        """ Project a 2D image to 3D physical space within range x_range_mm, y_range_mm using TPS interpolation."""
        # Input checks
        x_range_mm = np.array(x_range_mm)
        y_range_mm = np.array(y_range_mm)

        x_range_px = x_range_mm / pixel_size_mm
        y_range_px = y_range_mm / pixel_size_mm

        # Define the destination grid 
        x_px = np.linspace(x_range_px[0], x_range_px[1], int(x_range_px[1]-x_range_px[0])+1)
        y_px = np.linspace(y_range_px[0], y_range_px[1], int((y_range_px[1] - y_range_px[0]))+1)
        xx, yy = np.meshgrid(x_px, y_px)

        # Map uv to xyz in mm
        uv_points = np.vstack([xx.ravel(), yy.ravel()]).T
        xyz_points = self.tps_interpolator(uv_points)
        mapped_u = xyz_points[:, 0].reshape(xx.shape)
        mapped_v = xyz_points[:, 1].reshape(xx.shape)

        mapped_u = (mapped_u) / pixel_size_mm
        mapped_v = (mapped_v ) / pixel_size_mm

        # Handle RGB images
        if len(cv2_image.shape) == 3:
            # Apply map_coordinates to each channel separately
            warped_channels = [
                map_coordinates(
                    cv2_image[:, :, channel],  # Extract one channel
                    [mapped_v, mapped_u],  # Use the same coordinates for all channels
                    order=1,  
                    mode='constant', 
                    cval=0.0 
                )
                for channel in range(cv2_image.shape[2])  # Loop over channels
            ]
            # Stack the warped channels back into a 3D image
            transformed_image = np.stack(warped_channels, axis=-1)
        else: # Grayscale
            transformed_image = map_coordinates(
                cv2_image,
                [mapped_v, mapped_u],  
                order=1, 
                mode='constant',  
                cval=0.0 
            )

        return transformed_image    
    