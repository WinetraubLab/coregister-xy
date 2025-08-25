import numpy as np
import matplotlib.pyplot as plt
import itertools

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

def core_PCR99a(xyz_gt, xyz_est, log_ratio_mat, sort_idx, n_hypo, thr1, sigma, thr2):
    """
    This function contains the Python adaptation of the original PCR99a algorithm.
    Returns:
        A, B: arrays of corresponding inlier point pairs.
    """
    n = xyz_gt.shape[1]

    # Preallocate hypothesis matrices
    Sx_mat = np.zeros((5, n_hypo))
    Sy_mat = np.zeros((5, n_hypo))
    Sz_mat = np.zeros((5, n_hypo))

    max_max_nInliers = 0
    c = 0
    idx_inliers = []

    triplets = list(itertools.combinations(range(n), 3))

    for t in triplets:
        # print(t)
        i,j,k = t
                    

        if k < 0 or k >= n:
            continue  # skip invalid

        # Map to original indices
        i_old = sort_idx[i]
        j_old = sort_idx[j]
        k_old = sort_idx[k]

        # --- Prescreening ---
        log_ratio_ij = log_ratio_mat[i_old, j_old]
        log_ratio_jk = log_ratio_mat[j_old, k_old]
        log_ratio_ki = log_ratio_mat[k_old, i_old]

        e1 = abs(log_ratio_ij - log_ratio_jk)
        e2 = abs(log_ratio_jk - log_ratio_ki)
        e3 = abs(log_ratio_ki - log_ratio_ij)

        if e1 > thr1 or e2 > thr1 or e3 > thr1:
            continue

        c += 1
        A = xyz_gt[:, [i, j, k]]
        B = xyz_est[:, [i, j, k]]

        scale, R, t = sRt_from_N_points(A, B)

        if np.any(np.isnan([scale, *R.flatten(), *t])):
            # Skip degenerate triple
            continue

        # Fill hypothesis matrices (5×n_hypo)
        Sx_mat[:, c - 1] = [scale*R[0,0], scale*R[0,1], scale*R[0,2], t[0], 1]
        Sy_mat[:, c - 1] = [scale*R[1,0], scale*R[1,1], scale*R[1,2], t[1], 1]
        Sz_mat[:, c - 1] = [scale*R[2,0], scale*R[2,1], scale*R[2,2], t[2], 1]

        # When batch is full, evaluate inliers
        if c == n_hypo:
            c = 0

            Ex = np.hstack([
                -xyz_gt.T,
                -np.ones((n, 1)),
                xyz_est[0, :].reshape(-1, 1)
            ]) @ Sx_mat
            Ey = np.hstack([
                -xyz_gt.T,
                -np.ones((n, 1)),
                xyz_est[1, :].reshape(-1, 1)
            ]) @ Sy_mat
            Ez = np.hstack([
                -xyz_gt.T,
                -np.ones((n, 1)),
                xyz_est[2, :].reshape(-1, 1)
            ]) @ Sz_mat

            E = Ex**2 + Ey**2 + Ez**2
            e_thr = (sigma * thr2) ** 2

            nInliers = np.sum(E <= e_thr, axis=0)
            max_nInliers = np.max(nInliers)
            idx = np.argmax(nInliers)

            if max_max_nInliers < max_nInliers:
                max_max_nInliers = max_nInliers
                idx_inliers = np.where(E[:, idx] <= e_thr)[0]

                if max_max_nInliers >= max(9, round(n*0.009)):
                    break

    if len(idx_inliers) == 0:
        print("No inliers")
        return np.nan, np.nan, np.nan, []

    A = xyz_gt[:,idx_inliers]
    B = xyz_est[:,idx_inliers]

    return A, B

def run_PCR99a():
    # 1. Pairwise squared distance, log ratio matrix

    # 2. Score correspondence pairs

    # 3. RANSAC round 1
    pass
