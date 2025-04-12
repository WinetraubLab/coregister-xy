import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
import numpy.testing as npt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class FitPlaneElastic:
    """
    A class to perform 2D-to-3D Thin Plate Spline (TPS) transformations using RBFInterpolator.
    Supports forward mapping (uv -> xyz) and reverse mapping (xyz -> uv) using a separate interpolator.
    """
    
    def __init__(self,
                 anchor_points_uv_pix=None,
                 anchor_points_xyz_mm=None,
                 uv_to_xyz_elastic_interpolator=None, xyz_to_uv_elastic_interpolator=None, normal=None):
        """
        Initialize the FitPlaneElastic class.

        Args:
            uv_to_xyz_elastic_interpolator: An RBFInterpolator object for forward mapping (uv -> xyz).
            xyz_to_uv_elastic_interpolator: An RBFInterpolator object for reverse mapping (xyz -> uv).
            anchor_points_uv_pix: Store anchor_points for future usage.
            anchor_points_xyz_mm: Store anchor_points for future usage.
        """
        self.uv_to_xyz_elastic_interpolator = uv_to_xyz_elastic_interpolator  # Forward interpolator (uv -> xyz)
        self.xyz_to_uv_elastic_interpolator = xyz_to_uv_elastic_interpolator  # Inverse interpolator (xyz -> uv)
        self.anchor_points_xyz_mm = anchor_points_xyz_mm
        self.anchor_points_uv_pix = anchor_points_uv_pix
        self.norm = normal

        # Fit a linear version
        if self.uv_to_xyz_elastic_interpolator is not None:
            self.uv_to_xyz_affine_interpolator = LinearRegression(fit_intercept=True)
            self.uv_to_xyz_affine_interpolator.fit(anchor_points_uv_pix, anchor_points_xyz_mm)
    
    @classmethod
    def from_points(cls, anchor_points_uv_pix, anchor_points_xyz_mm, smoothing=0, print_inputs=False):
        """
        Initialize a FitPlaneElastic object using control points.

        Args:
            anchor_points_uv_pix: These are the positions of anchor points on the image (uv) as a numpy array of shape (n, 2).
            anchor_points_xyz_mm: These are the same points in physical space (xyz) as a numpy array of shape (n, 3).
            smoothing: Smoothing parameter. The interpolator perfectly fits the data when this is set to 0. 
            Larger values result in more regularization and a more relaxed fit. Recommended value range: 1e-6 to 1 (start small)
            print_inputs: If True, print the inputs for debugging.

        Returns:
            A FitPlaneElastic object.
        """
        # Input validation
        anchor_points_uv_pix = np.array(anchor_points_uv_pix)
        anchor_points_xyz_mm = np.array(anchor_points_xyz_mm)
        if anchor_points_uv_pix.shape[0] != anchor_points_xyz_mm.shape[0]:
            raise ValueError("Number of points should be the same between anchor_points_uv_pix and anchor_points_xyz_mm")
        if anchor_points_uv_pix.shape[1] != 2:
            raise ValueError("anchor_points_uv_pix must have shape (n, 2)")
        if anchor_points_xyz_mm.shape[1] != 3:
            raise ValueError("anchor_points_xyz_mm must have shape (n, 3)")

        # Print inputs for debugging
        if print_inputs:
            print("anchor_points_uv_pix:\n", anchor_points_uv_pix)
            print("anchor_points_xyz_mm:\n", anchor_points_xyz_mm)

        # Create forward interpolator (uv -> xyz)
        uv_to_xyz_elastic_interpolator = RBFInterpolator(
            anchor_points_uv_pix,  # 2D source points (uv)
            anchor_points_xyz_mm,  # 3D target points (xyz)
            kernel='thin_plate_spline',  
            neighbors=None,  # Use all points for interpolation
            smoothing=smoothing
        )

        # Inverse mapping
        # Prevent singularity
        perturbed_anchor_points_xyz_mm = anchor_points_xyz_mm + np.random.normal(scale=1e-12, size=anchor_points_xyz_mm.shape)
        
        xyz_to_uv_elastic_interpolator = RBFInterpolator(
            perturbed_anchor_points_xyz_mm[:,:2],  # Use only x and y for inverse (2D)
            anchor_points_uv_pix,
            kernel='thin_plate_spline', 
            neighbors=None,
            smoothing=smoothing
        )

        # Check that this mapping works x = reverse(forward(x))
        test_uv = xyz_to_uv_elastic_interpolator(anchor_points_xyz_mm[:, :2])
        test_xyz = uv_to_xyz_elastic_interpolator(test_uv)
        distance_error_mm = np.linalg.norm((test_xyz - anchor_points_xyz_mm), axis=1)
        if np.any(distance_error_mm > 1e-3): # Consistency under 1 micron is okay!
            raise AssertionError(
                "Inverse consistency check failed. Check that the anchor points are not in an evenly spaced grid, or reduce smoothing parameter."
            )
        
        def normal(xyz_mm):
            """ Uses SVD to find the normal vector of the best fit plane for the provided XYZ (template) points.
            """

            # Subtract the centroid to center the points
            centroid = np.mean(xyz_mm, axis=0) 
            centered_points = xyz_mm - centroid

            # SVD
            _, _, vh = np.linalg.svd(centered_points)

            # The last row of vh is the normal vector to the best-fit plane
            normal_vector = vh[-1, :]  

            # Normalize the normal vector 
            normal_vector /= np.linalg.norm(normal_vector)
            if normal_vector[2] < 0:
                normal_vector *= -1 # positive direction

            return normal_vector
        
        norm = normal(anchor_points_xyz_mm)

        return cls(anchor_points_uv_pix, anchor_points_xyz_mm, uv_to_xyz_elastic_interpolator, xyz_to_uv_elastic_interpolator, norm)
    
    def get_xyz_from_uv(self, uv_pix):
        """
        Map 2D uv coordinates to 3D xyz coordinates using the forward interpolator.

        Args:
            uv_pix: 2D uv coordinates as a numpy array of shape (n, 2).

        Returns:
            3D xyz coordinates as a numpy array of shape (n, 3), units are mm.
        """
        uv_pix = np.array(uv_pix)
        if uv_pix.ndim == 1:
            uv_pix = uv_pix[np.newaxis, :]  # Add batch dimension for single point
        return self.uv_to_xyz_elastic_interpolator(uv_pix)

    def get_xyz_from_uv_affine(self, uv_pix):
        """
        Transforms UV points to XYZ using affine transformation.

        Args:
            uv_pix: 2D uv coordinates as a numpy array of shape (n, 2).

        Returns:
            3D xyz coordinates as a numpy array of shape (n, 3), units are mm.
        """
        uv_pix = np.array(uv_pix)
        if uv_pix.ndim == 1:
            uv_pix = uv_pix[np.newaxis, :]  # Add batch dimension for single point
        return self.uv_to_xyz_affine_interpolator.predict(uv_pix)
    
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
        return self.xyz_to_uv_elastic_interpolator(xyz_mm[:, :2])  # Use only x and y for inverse
    
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
                    order=3,                  # Cubic interpolation
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
                order=3,
                mode='constant',
                cval=0.0
            )

        return transformed_image

    def _split_vector_to_in_plane_and_out_plane(self, vec_xyz_mm):
        """
        Given a vector, split it into plane and out-plane components.
        Args:
            vec_xyz_mm: 3D xyz coordinates as a numpy array of shape (n, 3).
        Outputs:
            in_plane: 3D xyz coordinates as a numpy array of shape (n, 3).
            out_plane: 3D xyz coordinates as a numpy array of shape (n, 2).
        """
        vec_xyz_mm = np.array(vec_xyz_mm)
        if vec_xyz_mm.ndim == 1:
            flatten_output = True
            vec_xyz_mm = vec_xyz_mm[np.newaxis, :]
        else:
            flatten_output = False

        normal_repeated = np.tile(self.norm.reshape(1, -1), (vec_xyz_mm.shape[0], 1))

        # Project vector on normal direction to get the out of plane direction
        out_plane_mm = np.sum(
            vec_xyz_mm * normal_repeated, axis=1, keepdims=True) * normal_repeated

        # In plane is what is left
        in_plane_mm = vec_xyz_mm - out_plane_mm

        if flatten_output:
            in_plane_mm = in_plane_mm.flatten()
            out_plane_mm = out_plane_mm.flatten()

        return in_plane_mm, out_plane_mm

    def get_elastic_affine_diff_mm(self, uv_pix):
        """
            Computes the difference between elastic and affine transformation, split to in plane and out-plane.
        """
        xyz_elastic = self.get_xyz_from_uv(uv_pix)
        xyz_affine = self.get_xyz_from_uv_affine(uv_pix)
        return self._split_vector_to_in_plane_and_out_plane(xyz_elastic - xyz_affine)
    
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
        assert uv_pix.shape[0] == xyz_mm.shape[0], "Mismatch in number of UV and XYZ points"

        uv_to_xyz = np.squeeze(np.array([self.get_xyz_from_uv(p) for p in uv_pix]))
        # Error vector
        error_xyz_mm = xyz_mm - uv_to_xyz
        normal = self.norm.reshape(1, -1) 
        normal_repeated = np.tile(normal, (xyz_mm.shape[0], 1))

        # Project error on normal direction
        error_xyz_projected_on_normal_mm = np.sum(error_xyz_mm * normal_repeated, axis=1, keepdims=True) * normal_repeated
        # Out of plane error is in direction of the normal
        out_plane_error_mm = np.linalg.norm(error_xyz_projected_on_normal_mm, axis=1)

        # Overall error
        all_error_mm = np.linalg.norm(error_xyz_mm, axis=1)
        in_plane_error_mm = np.sqrt(all_error_mm**2 - out_plane_error_mm**2)

        if mean:
            return np.mean(in_plane_error_mm), np.mean(out_plane_error_mm)
        else:
            return in_plane_error_mm, out_plane_error_mm

    def plot_explore_anchor_points_fit_quality(self, figure_title=""):
        """
                Plot how well the plane fit matches anchor points
                figure_title: figure title if exists.
        """

        # Convert UV points to XYZ
        plane_fit_xyz_mm = np.array([self.get_xyz_from_uv(t) for t in self.anchor_points_uv_pix]).squeeze()

        # Set up  figure
        fig, axes = plt.subplots(1, 2, figsize=(4.5 *2, 4.5), constrained_layout=True)

        # Plot XY Projection
        axes[0].scatter(
            plane_fit_xyz_mm[:, 0], plane_fit_xyz_mm[:, 1], label="Anchor Points (With Fit)")
        axes[0].scatter(
            self.anchor_points_xyz_mm[:, 0], self.anchor_points_xyz_mm[:, 1],
            label="Anchor Points (Raw)", marker='^')
        for pf_xyz, ap_xyz in zip(plane_fit_xyz_mm, self.anchor_points_xyz_mm):
            axes[0].plot([pf_xyz[0], ap_xyz[0]], [pf_xyz[1], ap_xyz[1]], c='k')
        axes[0].set_xlabel("X [mm]")
        axes[0].set_ylabel("Y [mm]")
        axes[0].grid(True)
        axes[0].legend( loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
        axes[0].set_title("XY Projection of Anchor Points\n", fontsize=14)

        # Plot XZ Projection
        axes[1].scatter(
            plane_fit_xyz_mm[:, 0], plane_fit_xyz_mm[:, 2], label="Anchor Points (With Fit)")
        axes[1].scatter(
            self.anchor_points_xyz_mm[:, 0], self.anchor_points_xyz_mm[:, 2],
            label="Anchor Points (Raw)", marker='^')
        for pf_xyz, ap_xyz in zip(plane_fit_xyz_mm, self.anchor_points_xyz_mm):
            axes[0].plot([pf_xyz[0], ap_xyz[0]], [pf_xyz[2], ap_xyz[2]], c='k')
        axes[1].set_xlabel("X [mm]")
        axes[1].set_ylabel("Z [mm]")
        axes[1].grid(True)
        axes[1].set_title("XZ Projection of Anchor Points", fontsize=14)

        fig.suptitle(figure_title, fontsize=14)
        plt.show()

        