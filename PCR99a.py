import numpy as np
import matplotlib.pyplot as plt

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
