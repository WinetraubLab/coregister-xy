from cellpose import io, models, plot
import matplotlib.pyplot as plt
import cv2
import numpy as np

from skimage.color import rgb2gray
from skimage.measure import regionprops, label
from scipy.ndimage import mean as nd_mean, binary_dilation
from skimage.morphology import disk
from skimage.transform import downscale_local_mean

def _clahe_normalize(img):
    """
    Performs CLAHE normalization on a 2D input image. 
    Enhances local contrast by applying histogram equalization to small tiles.
    Inputs:
        img: a 2D image.
    Returns:
        a normalized 2D image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    if img.ndim == 3 and img.shape[2] == 3:
        # CLAHE on L channel
        img = img.astype(np.uint8) 
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        # Grayscale 
        gray = img if img.ndim == 2 else img.squeeze()
        gray = (gray * 255).astype(np.uint8) if gray.dtype != np.uint8 else gray
        return clahe.apply(gray)

    else:
        raise ValueError(f"Unsupported image shape or type: {img.shape}, dtype: {img.dtype}")

def _global_normalization(image):
    """
    Apply global luminance-channel normalization on a 2D image.
    Inputs:
        image: 2D image.
    """
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image

def _filter_masks_by_darkness(image, masks, ring_size=3, min_diff=0, keep_brighter_than=False):
    """
    Filter cell masks by brightness/contrast.
    Inputs:
        image: 3D or 2D image.
        masks: 3D or 2D volume where values greater than 0 represent detected cell masks.
        ring_size: region border thickness in px to consider when comparing brightness
        min_diff: brightness difference threshold to keep cell masks
        keep_brighter_than: False = keep cells darker than surrounding tissue. True = keep cells brighter than surrounding tissue.
    Returns:
        Volume: same shape as masks, where values greater than 0 represent filtered cell masks.
    """
    # Convert RGB to grayscale
    if image.ndim == 3 and image.shape[-1] == 3:
        image = rgb2gray(image)

    if image.shape != masks.shape:
        raise ValueError("Image and mask shapes must match after grayscale conversion")

    # If 2D, expand to 3D for uniformity
    if image.ndim == 2:
        image = image[np.newaxis, ...]
        masks = masks[np.newaxis, ...]

    footprint = disk(ring_size)
    filtered = np.zeros_like(masks)
    rem = 0

    for z in range(image.shape[0]):
        img_slice = image[z]
        mask_slice = masks[z]

        labels = np.unique(mask_slice)
        labels = labels[labels != 0]
        if len(labels) == 0:
            continue

        # Inside means
        inside_means = nd_mean(img_slice, labels=mask_slice, index=labels)

        # Outside means
        dilated = binary_dilation(mask_slice > 0, structure=footprint)
        border_mask = dilated & (mask_slice == 0)

        # Assign border pixels to nearest mask via dilation labeling
        from scipy.ndimage import grey_dilation
        border_labels = grey_dilation(mask_slice, footprint=footprint)
        border_labels[~border_mask] = 0

        outside_means = nd_mean(img_slice, labels=border_labels, index=labels)
        diffs = inside_means - outside_means
        if keep_brighter_than:
            keep = diffs > min_diff
        else:
            keep = diffs < -min_diff

        # Assign kept masks with new sequential IDs
        for lab, k in zip(labels, keep):
            if k:
                filtered[z][mask_slice == lab] = lab
            else:
                rem += 1

    # Squeeze back
    if filtered.shape[0] == 1:
        filtered = filtered[0]

    print(f"{rem} cell masks removed.")
    return filtered

def segment_cells(image, avg_cell_diameter, flow_threshold=0.85, cellprob_threshold=-8, keep_dark_cells=True, gpu=True, normalization="global"):
    """
    Segment cells from 2D or 3D image.
    Inputs:
        image: the pre-cropped 2D or 3D image to be segmented.
        avg_cell_diameter: expected average diameter of cells in the image.
        flow_threshold: controls how strictly the predicted flow fields follow expected cell behavior. 
                Increase to allow more cell detections. 
                Typical range 0-1 (recommend 0.8-0.85 for OCT, 0.7 for fluorescent/histology)
        cellprob_threshold: minimum threshold applied to the predicted cell probability map to detect cells.
                Decrease to allow more cell detections. 
                Typical range -10 to 10 (recommend -8 to -9 for OCT, -5 for fluorescent/histology)
        keep_dark_cells: if True, keeps only cells whose insides are darker than their immediate surroundings.
        gpu: whether GPU is available for acceleration. True/False (highly recommend GPU)
        normalization: choose CLAHE (local) or global image normalization. set to 'clahe' or 'global' or 'none'
    Returns:
        filtered_masks
    """

    if image.dtype != np.uint8:
        image_norm = (image - image.min()) / (image.max() - image.min())
        image = image_norm

    if normalization == 'clahe': # local normalization
        if image.ndim == 2:
            image = _clahe_normalize(image)
        elif image.ndim == 3:
            image = np.array([_clahe_normalize(o) for o in image])
            image = np.array([cv2.GaussianBlur(o, (5, 5), 0) for o in image])

    elif normalization == 'global': # global normalization
        if image.ndim == 2:
            image = _global_normalization(image)
        if image.ndim == 3:
            image = np.array([_global_normalization(o) for o in image])

    model = models.Cellpose(model_type='cyto2', gpu=gpu)

    if image.ndim == 3:
        masks, flows, styles, diams = model.eval(
            image,
            diameter=avg_cell_diameter, 
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            stitch_threshold=0.3,
            channels=[0, 0]
        )
    if image.ndim == 2:
        masks, flows, styles, diams = model.eval(
            image,
            diameter=avg_cell_diameter, 
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0]
        )

    if keep_dark_cells:
        masks = _filter_masks_by_darkness(image, masks)

    return masks, flows
