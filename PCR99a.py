import numpy as np
import matplotlib.pyplot as plt

def sRt_from_3points(A, B):
    """
    Compute scale (s), rotation (R), translation (t) from 3 point pairs.
    A: (3,3) ground truth points
    B: (3,3) estimated points
    """
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    A_c = A - centroid_A
    B_c = B - centroid_B

    # Axes for B
    v12 = B_c[:, 1] - B_c[:, 0]
    x_axis = v12 / np.linalg.norm(v12)
    v13 = B_c[:, 2] - B_c[:, 0]
    v23 = np.cross(v12, v13)
    y_axis = v23 / np.linalg.norm(v23)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Axes for A
    v12 = A_c[:, 1] - A_c[:, 0]
    x_axis_ = v12 / np.linalg.norm(v12)
    v13 = A_c[:, 2] - A_c[:, 0]
    v23 = np.cross(v12, v13)
    y_axis_ = v23 / np.linalg.norm(v23)
    z_axis_ = np.cross(x_axis_, y_axis_)
    z_axis_ /= np.linalg.norm(z_axis_)

    R = np.column_stack([x_axis, y_axis, z_axis]) @ np.column_stack([x_axis_, y_axis_, z_axis_]).T

    # Scale
    num = sum(B_c[:, i].T @ R @ A_c[:, i] for i in range(A.shape[1]))
    den = sum(A_c[:, i].T @ A_c[:, i] for i in range(A.shape[1]))
    s = num / den

    # Translation
    t = centroid_B.flatten() - s * R @ centroid_A.flatten()

    return s, R, t

def sRt_from_N_points(A, B):
    """
    Compute scale (s), rotation (R), translation (t) from N point pairs.
    A: (3,n) ground truth points
    B: (3,n) estimated points
    """
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    A_c = A - centroid_A
    B_c = B - centroid_B

    U, _, Vh = np.linalg.svd(A_c @ B_c.T)
    V = Vh.T
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ U.T))]) @ U.T

    num = sum(B_c[:, i].T @ R @ A_c[:, i] for i in range(A.shape[1]))
    den = sum(A_c[:, i].T @ A_c[:, i] for i in range(A.shape[1]))
    s = num / den

    t = centroid_B.flatten() - s * R @ centroid_A.flatten()

    return s, R, t
