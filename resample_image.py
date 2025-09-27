import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def resample_linear(oct_volume, T, image_shape=None, tile_size=1000, verbose=False):
    """
    Sample the OCT volume along a transformed 2D plane.

    Args:
        oct_volume: np.ndarray of shape (Z, Y, X)
        T: 4x4 affine matrix (image to 3D space)
        image_shape: tuple (H, W): size of the image to sample from OCT volume. Defaults to same (H,W) 
                size as the OCT image.
        tile_size: side length of each tile in px (default: 1000). Decrease this number if not enough RAM to process full image.
    Returns:
        warped_img: (H, W) 2D image of sampled values
    """
    if image_shape == None:
        image_shape = oct_volume[0].shape[1:]
    else:
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
                np.full_like(xx.ravel(), 1),
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

def resample_linear_minimal_projection(thickness, oct_volume, T, image_shape=None,
                     transform_order="xyz", volume_order="zyx", plane_axes=("x","y"), verbose=False):
    """
    Sample minimal projection of OCT along a transformed plane, with slab thickness.
    Inputs:
        thickness: Thickness of slab, in px
        oct_volume: np.ndarray of shape (Z, Y, X)
        T: 4x4 affine matrix (image to 3D space)
        image_shape: tuple (H, W): size of the image to sample from OCT volume. Defaults to same (H,W) 
                size as the OCT image.
    """
    warped_imgs = []

    if image_shape == None:
        image_shape = oct_volume[0].shape[1:]
    for i in range(thickness):
        img, zc = resample_linear(oct_volume, T, image_shape, 1 + i-(thickness // 2), verbose=verbose)
        warped_imgs.append(img)
    arr = np.array(warped_imgs)
    return np.min(arr, axis=0)