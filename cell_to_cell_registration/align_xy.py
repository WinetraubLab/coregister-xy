import numpy as np
from scipy.ndimage import map_coordinates, spline_filter, affine_transform
from scipy.interpolate import Rbf

def elastic_warp_image_2d_and_points(image, source_pts, dest_pts, order=3, output_shape=None):
    """
    Warp an image using B-spline interpolation from control points.

    Inputs:
        image (H, W): Grayscale image
        source_pts: Source control points (y, x). shape (n,2)
        dest_pts: Destination control points (y, x). shape (n,2)
        order: Spline interpolation order (default=3 for cubic)
        points_to_warp: Points to transform, from the coordinate space of source_pts
        output_shape (H_out, W_out): Shape of output image (optional)

    Returns:
        Warped image (H_out, W_out) or (H, W) if output_shape not given
    """
    is_color = image.ndim == 3 and image.shape[2] == 3

    H, W = image.shape[:2]
    if output_shape is None:
        H_out, W_out = H, W
    else:
        H_out, W_out = output_shape

    grid_y, grid_x = np.mgrid[0:H_out, 0:W_out]

    # Compute displacement vectors at control points
    dY = dest_pts[:, 0] - source_pts[:, 0]
    dX = dest_pts[:, 1] - source_pts[:, 1]

    # Fit RBF interpolators
    interp_dY = Rbf(source_pts[:, 1], source_pts[:, 0], dY, function='thin_plate')
    interp_dX = Rbf(source_pts[:, 1], source_pts[:, 0], dX, function='thin_plate')

    # Interpolate displacement over the output grid
    disp_y = interp_dY(grid_x, grid_y)
    disp_x = interp_dX(grid_x, grid_y)
    sample_y = grid_y + disp_y
    sample_x = grid_x + disp_x

    # Smooth image
    if is_color:
        warped_channels = []
        for c in range(3):
            image_smooth_c = spline_filter(image[:, :, c], order=order)
            warped_c = map_coordinates(image_smooth_c, [sample_y, sample_x], order=order, mode='constant')
            warped_channels.append(warped_c)

        warped_image = np.stack(warped_channels, axis=-1)
    else:
        image_smooth = spline_filter(image, order=order)
        warped_image = map_coordinates(image_smooth, [sample_y, sample_x], order=order, mode='constant')

    return warped_image

def affine_warp_image_2d_and_points(image, source_pts, dest_pts, order=3, output_shape=None):
    """
    Warp the input 2D image using affine transform matrix.

    Inputs:
        image: 2D numpy array (H, W)
        T: 3x3 affine transform matrix from _compute_affine_yx
        output_shape: (H_out, W_out)
        points_to_warp: points to transform, from the coordinate space of source_pts
        order: interpolation order

    Returns:
        Warped image as 2D numpy array
    """
    if output_shape is None:
        output_shape = image.shape

    T = _compute_affine_yx(source_pts, dest_pts)
    matrix = T[:2, :2]
    offset = T[:2, 2]

    # Invert because affine_transform uses inverse mapping
    matrix_inv = np.linalg.inv(matrix)
    offset_inv = -matrix_inv @ offset

    warped = affine_transform(
        image,
        matrix_inv,
        offset=offset_inv,
        output_shape=output_shape,
        order=order,
        mode='constant'
    )

    return warped

def _compute_affine_yx(A, B):
    """
    Computes the 2D affine transformation matrix T that maps points A to points B.
    Inputs:
        A: (N, 2) array of source points (y, x)
        B: (N, 2) array of destination points (y, x)
    Returns:
        T: 3x3 affine transformation matrix
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required for an affine transformation.")
    # x,y
    A_xy = A[:, ::-1]  # (N, 2)
    B_xy = B[:, ::-1]

    A_h = np.hstack([A_xy, np.ones((A_xy.shape[0], 1))])
    T, _, _, _ = np.linalg.lstsq(A_h, B_xy, rcond=None)  # T (3, 2)
    T = T.T 
    T_full = np.vstack([T, [0, 0, 1]])
    return T_full
