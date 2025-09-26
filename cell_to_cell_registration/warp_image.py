import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates, spline_filter
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from scipy.ndimage import affine_transform

def sample_oct_from_plane(oct_volume, T, image_shape, hist_assigned_z=1, tile_size=500, verbose=False):
    """
    Sample the OCT volume along a transformed 2D plane.

    Args:
        oct_volume: np.ndarray of shape (Z, Y, X)
        T: 4x4 affine matrix (image to 3D space)
        image_shape: tuple (H, W) of the image grid
        hist_assigned_z: initial assumed z coordinate of the flat histology image
        tile_size: size of each tile (default: 500)
    Returns:
        warped_img: (H, W) 2D image of sampled values
    """
    H, W = image_shape
    warped_img = np.zeros((H, W), dtype=np.float32)
    z_coords = np.zeros((H, W), dtype=np.float32)

    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            h, w = y1 - y0, x1 - x0

            # Create meshgrid for this tile
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
            ones = np.ones_like(xx)
            pixels_h = np.stack([
                xx.ravel(), yy.ravel(),
                np.full_like(xx.ravel(), hist_assigned_z),
                ones.ravel()
            ], axis=0)  # shape: (4, N)

            # Apply affine
            pts3D = T @ pixels_h
            pts3D = pts3D[:3] / pts3D[3]  # Normalize homogeneous coords

            # OCT volume is (Z, Y, X), so order: z, y, x
            coords = np.stack([pts3D[2], pts3D[1], pts3D[0]], axis=0)  # shape: (3, N)

            # Interpolate
            sampled_vals = map_coordinates(oct_volume, coords, order=3, mode='nearest')
            warped_img[y0:y1, x0:x1] = sampled_vals.reshape(h, w)
            z_coords[y0:y1, x0:x1] = coords[0].reshape(h, w)

    if verbose:
        plt.imshow(z_coords, cmap='viridis')
        plt.colorbar(label='Z Coordinate')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Z Coordinates of Sampled Points")
        plt.show()

        print("Z range of sampled points:", np.min(z_coords), np.max(z_coords))
        print("Y range of sampled points:", np.min(coords[1]), np.max(coords[1]))
        print("X range of sampled points:", np.min(coords[2]), np.max(coords[2]))

    return warped_img, z_coords

def minimal_projection(thickness, oct_volume, T, image_shape, hist_assigned_z=1,
                     transform_order="xyz", volume_order="zyx", plane_axes=("x","y"), verbose=False):
    """
    Sample minimal projection of OCT along a transformed plane, with slab thickness.
    Inputs:
        thickness: Thickness of slab, in px
        oct_volume: np.ndarray of shape (Z, Y, X)
        T: 4x4 affine matrix (image to 3D space)
        image_shape: tuple (H, W) of the image grid
        hist_assigned_z: initial assumed z coordinate of the flat histology image
    """
    warped_imgs = []
    for i in range(thickness):
        img, zc = sample_oct_from_plane(oct_volume, T, image_shape, hist_assigned_z + i-(thickness // 2), verbose=verbose)
        warped_imgs.append(img)
    arr = np.array(warped_imgs)
    return np.min(arr, axis=0)

def bspline_warp_image_2d(image, source_pts, dest_pts, order=3, output_shape=None):
    """
    Warp an image using B-spline interpolation from control points.

    Inputs:
        image (H, W): Grayscale image
        source_pts: Source control points (y, x). shape (n,2)
        dest_pts: Destination control points (y, x). shape (n,2)
        order: Spline interpolation order (default=3 for cubic)
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

def affine_warp_image_2d(image, source_pts, dest_pts, order=3, output_shape=None):
    """
    Warp the input 2D image using affine transform matrix.

    Inputs:
        image: 2D numpy array (H, W)
        T: 3x3 affine transform matrix from _compute_affine_yx
        output_shape: (H_out, W_out)
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
