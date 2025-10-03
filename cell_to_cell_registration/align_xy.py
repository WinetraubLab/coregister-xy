import numpy as np
from scipy.ndimage import map_coordinates, spline_filter, affine_transform
from scipy.interpolate import Rbf

def _elastic_warp_image_2d(image, source_pts, dest_pts, order=3, output_shape=None):
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

def _affine_warp_image_2d(image, source_pts, dest_pts, order=3, output_shape=None):
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

def align_and_crop_histology_image(histology_image, oct_coords_mm, histology_coords_px, 
                                   oct_crop_region_mm, align_mode='affine'):
    """
    Aligns a histology image to OCT coordinate space using point correspondences, and crops a region of interest defined in OCT space.
    This function assumes that the OCT coordinate system has an inverted y-axis 
    compared to the image coordinate system. 
  
    Inputs:
    ----------
    oct_coords_mm : (N, 3) array
        Coordinates of corresponding points in the OCT system. Units: mm (x,y,z).
        The Z-coordinate may be NaN if unable to determine Z from the fluorescent barcode at that point.
    
    histology_coords_px : (N, 2) array
        Coordinates of the corresponding points in the histology image. Units: px (i,j).
    
    oct_crop_region_mm : (4, 2) array
        The corner points of the region of interest to crop, specified in the OCT coordinate system. 
        Units: mm (x,y).

    histology_image : (H,W) or (H,W,C) array 
        The full histology image as a NumPy array. Can be 2D (grayscale) or 3D (RGB).

    align_mode: 'affine' or 'elastic'
        Specify whether to use affine or elastic rough alignment. Recommend affine unless you have 
        highly visible barcodes and multiple points per barcode.

    Returns:
    -------
    cropped_histology_image : ndarray
        The cropped region from the histology image. 1px/um scale.
    """
    # Convert to um
    oct_coords_mm = np.array(oct_coords_mm)
    histology_coords_px = np.array(histology_coords_px)

    oct_xy_mm = oct_coords_mm[:, :2].copy()
    oct_xy_um = oct_xy_mm * 1000 

    crop_poly_mm = np.array(oct_crop_region_mm)
    crop_poly_um = crop_poly_mm * 1000

    # Shift dest coordinates to positive space for image warp
    shift_offset = -np.minimum(np.min(np.vstack([oct_xy_um, crop_poly_um]), axis=0), 0)
    oct_xy_um_shifted = oct_xy_um + shift_offset
    crop_poly_um_shifted = crop_poly_um + shift_offset

    # Output image size=crop region
    max_coords = np.max(crop_poly_um_shifted, axis=0)
    output_shape = tuple(np.ceil(max_coords[::-1]).astype(int))  # (H, W)

    # Warp the image
    if histology_image.ndim == 2:
        if align_mode == 'affine':
            warped_img = _affine_warp_image_2d(
                image=histology_image,
                source_pts=histology_coords_px,
                dest_pts=oct_xy_um_shifted,
                output_shape=output_shape
            )
        elif align_mode == 'elastic':
            warped_img = _elastic_warp_image_2d(
                image=histology_image,
                source_pts=histology_coords_px,
                dest_pts=oct_xy_um_shifted,
                output_shape=output_shape
            )
    else:
        # Warp each channel
        warped_channels = []
        for c in range(histology_image.shape[2]):
            if align_mode == 'affine':
                warped_c = _affine_warp_image_2d(
                    image=histology_image[:, :, c],
                    source_pts=histology_coords_px,
                    dest_pts=oct_xy_um_shifted,
                    output_shape=output_shape
                )
            elif align_mode == 'elastic':
                warped_c = _elastic_warp_image_2d(
                    image=histology_image[:, :, c],
                    source_pts=histology_coords_px,
                    dest_pts=oct_xy_um_shifted,
                    output_shape=output_shape
                )
            warped_channels.append(warped_c)
        warped_img = np.stack(warped_channels, axis=-1)

    # Crop 
    crop_min = np.floor(np.min(crop_poly_um_shifted, axis=0)).astype(int)
    crop_max = np.ceil(np.max(crop_poly_um_shifted, axis=0)).astype(int)

    y_min, x_min = crop_min[::-1]
    y_max, x_max = crop_max[::-1]

    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(warped_img.shape[0], y_max)
    x_max = min(warped_img.shape[1], x_max)

    cropped_histology_image = warped_img[y_min:y_max, x_min:x_max]

    return cropped_histology_image
