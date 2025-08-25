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

def _score_correspondences(log_ratio_mat, thr1):
        """
        Step 2 of PCR99a.
        Args:
            log_ratio_mat: (n, n) matrix
            thr1: clipping threshold

        Returns:
            min_costs: (n,) array of minimum costs per row
        """
        n = log_ratio_mat.shape[0]
        min_costs = np.full(n, np.inf)

        for i in range(n):
            lr_mat = log_ratio_mat[i, :]
            max_lr, min_lr = np.nanmax(lr_mat), np.nanmin(lr_mat)
            lr_range = max_lr - min_lr
            lr_step = lr_range / max(1, round(lr_range / 0.1))

            lr_candidates = np.arange(min_lr, max_lr + lr_step/2, lr_step)  # inclusive range

            # Broadcast: residuals shape = (num_candidates, n)
            r = np.abs(lr_mat[None, :] - lr_candidates[:, None])
            r = np.minimum(r, thr1)  # clip

            # Cost for each candidate = sum across columns
            costs = np.nansum(r, axis=1)
            min_costs[i] = np.nanmin(costs)
        return min_costs

def run_PCR99a():
    # 1. Pairwise squared distance, log ratio matrix

    # 2. Score correspondence pairs

    # 3. RANSAC round 1
    pass