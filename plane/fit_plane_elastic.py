import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
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
                 smoothing=0, print_inputs=False):
        """
        Initialize the FitPlaneElastic class.

        Args:
            anchor_points_uv_pix: These are the positions of anchor points on the image (uv) as a numpy array of shape (n, 2).
            anchor_points_xyz_mm: These are the same points in physical space (xyz) as a numpy array of shape (n, 3).
            smoothing: Smoothing parameter. The interpolator perfectly fits the data when this is set to 0.
                Larger values result in more regularization and a more relaxed fit. Recommended value range: 1e-6 to 1 (start small)
            print_inputs: If True, print the inputs for debugging.
        """

        # Input validation
        anchor_points_uv_pix = np.array(anchor_points_uv_pix)
        anchor_points_xyz_mm = np.array(anchor_points_xyz_mm)
        if anchor_points_uv_pix.shape[0] != anchor_points_xyz_mm.shape[0]:
            raise ValueError(
                "Number of points should be the same between anchor_points_uv_pix and anchor_points_xyz_mm")
        if anchor_points_uv_pix.shape[1] != 2:
            raise ValueError("anchor_points_uv_pix must have shape (n, 2)")
        if anchor_points_xyz_mm.shape[1] != 3:
            raise ValueError("anchor_points_xyz_mm must have shape (n, 3)")

        # Print inputs for debugging
        if print_inputs:
            print("anchor_points_uv_pix:\n", anchor_points_uv_pix)
            print("anchor_points_xyz_mm:\n", anchor_points_xyz_mm)

        # Store anchor points
        self.anchor_points_xyz_mm = anchor_points_xyz_mm
        self.anchor_points_uv_pix = anchor_points_uv_pix

        # Create forward interpolator (uv -> xyz)
        self.uv_to_xyz_elastic_interpolator = RBFInterpolator(
            anchor_points_uv_pix,  # 2D source points (uv)
            anchor_points_xyz_mm,  # 3D target points (xyz)
            kernel='thin_plate_spline',
            neighbors=None,  # Use all points for interpolation
            smoothing=smoothing
        )

        # Create inverse interpolator (xyz -> uv)
        def create_inverse_interpolator():
            # Add random noise to prevent singularity
            perturbed_anchor_points_xyz_mm = (
                    anchor_points_xyz_mm +
                    np.random.normal(scale=1e-12, size=anchor_points_xyz_mm.shape))

            # Since RBFInterpolator cannot map 3D -> 2D, we lower the dimensions by focusing on in plane vectors
            in_plane_anchor_points_xyz_mm, _= self._split_vector_to_in_plane_and_out_plane(
                perturbed_anchor_points_xyz_mm, output_coordinate_system='plane')

            return RBFInterpolator(
                in_plane_anchor_points_xyz_mm,
                anchor_points_uv_pix,
                kernel='thin_plate_spline',
                neighbors=None,
                smoothing=smoothing
            )
        self.xyz_to_uv_elastic_interpolator = create_inverse_interpolator()

        # Check that  mapping works x = reverse(forward(x))
        test_uv = self.get_uv_from_xyz(anchor_points_xyz_mm)
        test_xyz = self.get_xyz_from_uv(test_uv)
        distance_error_mm = np.linalg.norm((test_xyz - anchor_points_xyz_mm), axis=1)
        if np.any(distance_error_mm > 1e-3):  # Consistency under 1 micron is okay!
            raise AssertionError(
                "Inverse consistency check failed. Check that the anchor points are not in an evenly spaced grid, or reduce smoothing parameter."
            )

        # Fit a linear (affine) interpolator
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
        return cls(anchor_points_uv_pix, anchor_points_xyz_mm)

    def normal(self):
        """
        Return a unit vector in the direction of the normal
        Uses SVD to find the normal vector of the best fit plane for aoncor points.
        """
        xyz_mm = self.anchor_points_xyz_mm

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
            normal_vector *= -1  # positive direction

        return normal_vector

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
        assert(uv_pix.shape[1] == 2) # Make sure that shape of input is (n, 2)
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

        assert(xyz_mm.shape[1] == 3) # Make sure that shape of input is (n, 3)

        # Since RBFInterpolator cannot map 3D -> 2D, we lower the dimensions by focusing on in plane vectors
        in_plane_xyz_mm, _ = self._split_vector_to_in_plane_and_out_plane(xyz_mm, output_coordinate_system='plane')
        return self.xyz_to_uv_elastic_interpolator(in_plane_xyz_mm)
    
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
        mm_points = np.vstack([xx_mm.ravel(), yy_mm.ravel(), np.zeros_like(yy_mm.ravel())]).T

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

    def _split_vector_to_in_plane_and_out_plane(
            self, vec_xyz_mm, forced_plane_normal = None, output_coordinate_system='physical'):
        """
        Given a vector, split it into plane and out-plane components.
        Args:
            vec_xyz_mm: 3D xyz coordinates as a numpy array of shape (n, 3).
            forced_plane_normal: When set to 3D vector, will override plane normal to provided vector
            output_coordinate_system: When set to 'physical' (default) then in_plane, out_plane will be 3D vectors (x,y,z)
                When set to 'plane' then in_plane will be a 2D vector in plane coordinates, out_plane will be 1D vector
                depicting out of plane coordinate
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

        # Get the normal
        if forced_plane_normal is None:
            normal = self.normal()
        else:
            assert np.allclose(np.linalg.norm(forced_plane_normal),1)
            normal = np.array(forced_plane_normal)
        normal_repeated = np.tile(normal.reshape(1, -1), (vec_xyz_mm.shape[0], 1))

        # Project vector on normal direction to get the out of plane direction
        out_plane_mm = np.sum(
            vec_xyz_mm * normal_repeated, axis=1, keepdims=True) * normal_repeated

        # In plane is what is left
        in_plane_mm = vec_xyz_mm - out_plane_mm

        if output_coordinate_system == 'plane':
            # Out of plane is a dot product with normal
            out_plane_mm = np.sum(vec_xyz_mm * normal_repeated, axis=1, keepdims=True)
            out_plane_mm = np.squeeze(out_plane_mm)

            # In plane coordinate system
            axis_1 = np.array([1,0,0]) # X-axis is first
            axis_2 = -np.cross(axis_1, normal) # Conj
            axis_1_repeated = np.tile(axis_1.reshape(1, -1), (vec_xyz_mm.shape[0], 1))
            axis_2_repeated = np.tile(axis_2.reshape(1, -1), (vec_xyz_mm.shape[0], 1))

            # Compute projection to each axis
            in_plane_coord_1_mm = np.sum(in_plane_mm * axis_1_repeated, axis=1, keepdims=True)
            in_plane_coord_2_mm = np.sum(in_plane_mm * axis_2_repeated, axis=1, keepdims=True)

            # Concatenate
            in_plane_mm = np.squeeze(np.array([in_plane_coord_1_mm, in_plane_coord_2_mm]).transpose(), axis=0)

        # Post process output
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

    def plot_explore_anchor_points_fit_quality(
            self, figure_title="", coordinate_system='physical', use_elastic_fit=True):
        """
            Plot how well the plane fit matches anchor points

            Args:
                figure_title: figure title if exists.
                coordinate_system: can be 'physical' (default) or 'fit'. If using physical, will plot errors in XY and
                    XZ coordinates. If using fit, will plot errors in in_plane and out_plane.
                use_elastic_fit: set to True to use elastic fit (default) or false to use affine fit.
        """

        # Capture anchor points raw and fit
        if use_elastic_fit:
            plane_fit_xyz_mm = np.array([self.get_xyz_from_uv(t) for t in self.anchor_points_uv_pix]).squeeze()
        else:
            plane_fit_xyz_mm = np.array([self.get_xyz_from_uv_affine(t) for t in self.anchor_points_uv_pix]).squeeze()

        # Split coordinates to in plane and out of plane
        if coordinate_system == 'physical':
            plane_fit_xyz_mm = plane_fit_xyz_mm
            anchor_points_xyz_mm = self.anchor_points_xyz_mm
            normal_axis = [0, 0, 1] # Z
            conj_axis = [0, 1, 0] # Y
        else:
            in_p, out_p = self._split_vector_to_in_plane_and_out_plane(plane_fit_xyz_mm, output_coordinate_system='plane')
            plane_fit_xyz_mm = np.array([np.squeeze(in_p[:,0]), np.squeeze(in_p[:,1]), out_p]).transpose()

            in_p, out_p = self._split_vector_to_in_plane_and_out_plane(self.anchor_points_xyz_mm, output_coordinate_system='plane')
            anchor_points_xyz_mm = np.array([np.squeeze(in_p[:,0]), np.squeeze(in_p[:,1]), out_p]).transpose()

            normal_axis = self.normal()
            conj_axis = -np.cross(np.array([1,0,0]), normal_axis)  # Conj

        # Set up  figure
        fig, axes = plt.subplots(1, 4, figsize=(4.5 * 4, 4.5), constrained_layout=True)

        # Plot XY/In plane Projection
        plt.subplot(1, 4, 1)
        plt.scatter(
            plane_fit_xyz_mm[:, 0], plane_fit_xyz_mm[:, 1], label="Anchor Points (With Fit)")
        plt.scatter(
            anchor_points_xyz_mm[:, 0], anchor_points_xyz_mm[:, 1],
            label="Anchor Points (Raw)", marker='^')
        for pf_xyz, ap_xyz in zip(plane_fit_xyz_mm, anchor_points_xyz_mm):
            plt.plot([pf_xyz[0], ap_xyz[0]], [pf_xyz[1], ap_xyz[1]], c='k')
        if coordinate_system == 'physical':
            plt.xlabel("X [mm]")
            plt.ylabel("Y [mm]")
            plt.title("XY Projection of Anchor Points\n", fontsize=14)
        else:
            normal_axis
            plt.xlabel("X [mm]\n[1, 0, 0]")
            plt.ylabel(f"Conj Axis [mm]\n[{conj_axis[0]:.2f}, {conj_axis[1]:.2f}, {conj_axis[2]:.2f}]")
            plt.title("In Plane Projection of Anchor Points\n", fontsize=14)
        plt.grid(True)
        plt.legend( loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)


        # Plot XZ Projection
        plt.subplot(1, 4, 2)
        plt.scatter(
            plane_fit_xyz_mm[:, 0], plane_fit_xyz_mm[:, 2], label="Anchor Points (With Fit)")
        plt.scatter(
            anchor_points_xyz_mm[:, 0], anchor_points_xyz_mm[:, 2],
            label="Anchor Points (Raw)", marker='^')
        for pf_xyz, ap_xyz in zip(plane_fit_xyz_mm, anchor_points_xyz_mm):
            plt.plot([pf_xyz[0], ap_xyz[0]], [pf_xyz[2], ap_xyz[2]], c='k')
        if coordinate_system == 'physical':
            plt.xlabel("X [mm]")
            plt.ylabel("Z [mm]")
            plt.title("XZ Projection of Anchor Points\n", fontsize=14)
        else:
            plt.xlabel("X [mm]\n[1, 0, 0]")
            plt.ylabel(f"Normal Axis [mm]\n[{normal_axis[0]:.2f}, {normal_axis[1]:.2f}, {normal_axis[2]:.2f}]")
            plt.title("Out of Plane Projection of Anchor Points\n", fontsize=14)
        plt.grid(True)

        # Error Histogram
        error_vec = plane_fit_xyz_mm - anchor_points_xyz_mm
        in_plane_error = np.linalg.norm(error_vec[:,:2], axis=1)
        out_plane_error = np.squeeze(np.abs(error_vec[:, 2]))

        plt.subplot(1, 4, 3)
        plt.hist(in_plane_error*1000, bins=10, color='teal', alpha=0.7)
        if coordinate_system == 'physical':
            plt.xlabel('XY Error [um]')
            plt.title(f'Histogram XY Errors\n(mean={np.mean(in_plane_error*1000):.1f}um)', fontsize=14)
        else:
            plt.xlabel('In Plane Error [um]')
            plt.title(f'Histogram In-Plane Errors\n(mean={np.mean(in_plane_error*1000):.1f}um)', fontsize=14)
        plt.ylabel('Frequency')

        plt.subplot(1, 4, 4)
        plt.hist(out_plane_error * 1000, bins=10, color='teal', alpha=0.7)
        if coordinate_system == 'physical':
            plt.xlabel('Z Error [um]')
            plt.title(f'Histogram Z Errors\n(mean={np.mean(out_plane_error * 1000):.1f}um)', fontsize=14)
        else:
            plt.xlabel('In Out-Plane Error [um]')
            plt.title(f'Histogram Out-Plane Errors\n(mean={np.mean(out_plane_error * 1000):.1f}um)', fontsize=14)
        plt.ylabel('Frequency')

        fig.suptitle(figure_title, fontsize=14)
        plt.show()