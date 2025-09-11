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
        Inputs:
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

def _core_PCR99a(xyz_gt, xyz_est, log_ratio_mat, sort_idx, n_hypo, thr1, pcr99_inlier_thresh):
    n = xyz_gt.shape[1]

    # --- Configurable parameters ---
    early_stop_on = True                                    # early stopping
    min_inliers_to_trigger_stop = max(9, round(n * 0.01))   # min inlier ratio: 1%, must be more than 9 pts
    no_improvement_limit = 10                               # Max iterations with no better result
    e_thr = pcr99_inlier_thresh

    Sx_mat = np.zeros((5, n_hypo))
    Sy_mat = np.zeros((5, n_hypo))
    Sz_mat = np.zeros((5, n_hypo))

    max_max_nInliers = 0
    no_improvement_iters = 0
    c = 0
    idx_inliers = []

    triplets = list(itertools.combinations(range(n), 3))

    for t in triplets:
        i, j, k = t
        if k < 0 or k >= n:
            continue

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
            continue  # Skip bad triplet

        # Store hypothesis
        Sx_mat[:, c - 1] = [scale*R[0,0], scale*R[0,1], scale*R[0,2], t[0], 1]
        Sy_mat[:, c - 1] = [scale*R[1,0], scale*R[1,1], scale*R[1,2], t[1], 1]
        Sz_mat[:, c - 1] = [scale*R[2,0], scale*R[2,1], scale*R[2,2], t[2], 1]

        # Evaluate when batch is full
        if c == n_hypo:
            c = 0

            Ex = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[0, :].reshape(-1, 1)]) @ Sx_mat
            Ey = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[1, :].reshape(-1, 1)]) @ Sy_mat
            Ez = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[2, :].reshape(-1, 1)]) @ Sz_mat

            E = Ex**2 + Ey**2 + Ez**2

            nInliers = np.sum(E <= e_thr, axis=0)
            max_nInliers = np.max(nInliers)
            idx = np.argmax(nInliers)

            # --- Check if best so far ---
            if max_nInliers > max_max_nInliers:
                max_max_nInliers = max_nInliers
                idx_inliers = np.where(E[:, idx] <= e_thr)[0]
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            # --- Early stopping logic ---
            if (
                early_stop_on and 
                max_max_nInliers >= min_inliers_to_trigger_stop and 
                no_improvement_iters >= no_improvement_limit
            ):
                break

    print("Best hypothesis inliers:", max_max_nInliers, "/", n)

    # --- Final result ---
    if len(idx_inliers) == 0:
        print("No inliers found.")
        return np.array([]), np.array([])

    A = xyz_gt[:, idx_inliers]
    B = xyz_est[:, idx_inliers]

    return A, B

def plane_ransac(points_from_oct, points_from_hist, n_iter = 2000, 
                 plane_inlier_thresh=5, z_dist_thresh=4,
                 penalty_threshold=8, xy_translation_penalty_weight=1):
    """
    RANSAC round 2: plane-based inlier set refinement.
    Params:
        points_from_oct, points_from_hist: corresponding point pairs from initial PCR99 inlier selection.
        n_iter: RANSAC iterations to perform.
        plane_inlier_thresh: The maximum perpendicular distance from a candidate plane at which a point is 
            still considered an inlier during RANSAC iterations.
        z_dist_thresh: threshold on the distance of points to the final plane. 
            It defines which points are retained as the final inlier set.
        penalty_threshold: amount of XY translation between the two point sets that is acceptable 
            before incurring a score penalty.
        xy_translation_penalty_weight: scaling factor for how severely to penalize XY translation beyond penalty_threshold.
    Returns:
        oct_points_final, hist_points_final: Arrays containing corresponding point pairs for selected inliers.
    """

    n = points_from_oct.shape[1]
    best_plane_normal = None
    best_plane_point = None
    best_score = -np.inf

    for _ in range(n_iter):
        # Randomly select subset
        subset_idx = np.random.choice(n, 3, replace=False)
        A_subset = points_from_oct[:, subset_idx]

        # Fit plane to A_subset
        mean_subset = np.mean(A_subset, axis=1, keepdims=True)
        A_centered_subset = A_subset - mean_subset

        # SVD for plane normal (least variance direction)
        _, _, V = np.linalg.svd(A_centered_subset.T, full_matrices=False)
        plane_normal_candidate = V.T[:, -1].reshape(-1, 1) 

        # Normalize plane normal
        plane_normal_candidate /= np.linalg.norm(plane_normal_candidate)

        # Distances from all points in A to this plane
        vecs_to_plane = points_from_oct - mean_subset  
        dists = np.abs(plane_normal_candidate.T @ vecs_to_plane).flatten()

        # Count inliers within threshold
        inliers_candidate = np.where(dists <= plane_inlier_thresh)[0]
        num_inliers_candidate = len(inliers_candidate)

        # Penalize large xy translations
        A_sub = points_from_oct[:, inliers_candidate]
        B_sub = points_from_hist[:, inliers_candidate]

        if A_sub.shape[1] < 1:
            continue

        s_temp, R_temp, t_temp = sRt_from_N_points(A_sub, B_sub)

        xy_trans = np.linalg.norm(t_temp[:2])

        score = num_inliers_candidate - xy_translation_penalty_weight * max(0, xy_trans - penalty_threshold)

        # Update best if score improved
        if score > best_score:
            best_score = score
            best_plane_normal = plane_normal_candidate
            best_plane_point = mean_subset

    best_plane_normal = best_plane_normal / np.linalg.norm(best_plane_normal)

    # Z distance to the plane (signed projection)
    vecs_to_plane = points_from_oct - best_plane_point.reshape(-1, 1)  # shape (3×N)
    z_dists = np.abs(best_plane_normal.T @ vecs_to_plane).flatten()  # shape (N,)
    valid_mask = z_dists <= z_dist_thresh

    final_inliers = np.where(valid_mask)[0]
    A_final = points_from_oct[:, final_inliers]
    B_final = points_from_hist[:, final_inliers]

    return A_final, B_final

def plot_point_pairs(points_from_oct, points_from_hist, title="", save=False):
    """
    Visualize inlier point pairs.
    Inputs:
        points_from_oct: reference point set
        points_from_hist: candidate set to align
        title: title of the graph
        save: whether to save image to filesystem as {title}.png
    """
    x_gt, y_gt, z_gt = points_from_oct[0, :], points_from_oct[1, :], points_from_oct[2, :]
    x_est, y_est, z_est = points_from_hist[0, :], points_from_hist[1, :], points_from_hist[2, :]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # XY subplot
    axes[0].scatter(x_gt, y_gt, s=30, c='g', label='OCT')
    axes[0].scatter(x_est, y_est, s=30, c='r', label="Hist")
    for i in range(len(x_gt)):
        axes[0].plot([x_gt[i], x_est[i]], [y_gt[i], y_est[i]], 'k-', linewidth=0.5)
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    axes[0].set_title('XY')
    axes[0].grid(True)

    # XZ subplot
    axes[1].scatter(x_gt, z_gt, s=30, c='g', label='OCT')
    axes[1].scatter(x_est, z_est, s=30, c='r', label="Hist")
    for i in range(len(x_gt)):
        axes[1].plot([x_gt[i], x_est[i]], [z_gt[i], z_est[i]], 'k-', linewidth=0.5)
    axes[1].set_xlabel('X'); axes[1].set_ylabel('Z')
    axes[1].set_title('XZ')
    axes[1].grid(True)

    # YZ subplot
    axes[2].scatter(y_gt, z_gt, s=30, c='g', label='OCT')
    axes[2].scatter(y_est, z_est, s=30, c='r', label="Hist")
    for i in range(len(x_gt)):
        axes[2].plot([y_gt[i], y_est[i]], [z_gt[i], z_est[i]], 'k-', linewidth=0.5)
    axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z')
    axes[2].set_title('YZ')
    axes[2].grid(True)

    axes[2].legend()
    plt.suptitle(title)
    plt.show()
    
    if save:
        plt.savefig(f"{title}.png")

def _compute_affine(A, B):
    """
    Computes the 3D affine transformation matrix T (3x4) that maps points A to points B.
    Inputs:
        A: (3,N) array of source 3D points.
        B: (3,N) array of destination 3D points.
    Returns:
        T: 4x4 affine transformation matrix.
    """
    A = np.asarray(A).T
    B = np.asarray(B).T

    if A.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required for an affine transformation.")

    A_h = np.hstack([A, np.ones((A.shape[0], 1))])  # Nx4

    # Solve for the transformation matrix using least squares: A_h * T.T ≈ B_temp
    T, residuals, rank, s = np.linalg.lstsq(A_h, B, rcond=None)  # T is 4x3
    T = T.T  # 3x4
    T_4x4 = np.vstack([T, [0, 0, 0, 1]])

    return T_4x4

def calculate_affine_alignment(xyz_oct, xyz_hist, n_hypo=1000, thr1=0.03, pcr99_inlier_thresh=50, n_iter=2000, plane_inlier_thresh=5, z_dist_thresh=4,
                 penalty_threshold=8, xy_translation_penalty_weight=1):
    """
    Run full alignment algorithm. 
    Inputs:
        xyz_oct: Reference point set from 3D OCT segmentations, in pixels. (3,n)
        xyz_hist: Candidate set to align, from 2D histology image, in pixels. (3,n)
        n_hypo: Number of hypotheses to batch together before evaluating inliers. Smaller values are recommended for small point sets.
        thr1: Log-ratio consistency threshold for prescreening candidate triplets.
        pcr99_inlier_thresh: threshold for considering points inliers in PCR99a.
        n_iter: RANSAC iterations to perform.
        plane_inlier_thresh: The maximum perpendicular distance from a candidate plane at which a point is 
            still considered an inlier during RANSAC iterations.
        z_dist_thresh: threshold on the distance of points to the final plane. 
            It defines which points are retained as the final inlier set.
        penalty_threshold: amount of XY translation between the two point sets that is acceptable 
            before incurring a score penalty.
        xy_translation_penalty_weight: scaling factor for how severely to penalize XY translation beyond penalty_threshold.
    
    Returns:
    T: transformation matrix such that T @ A = B, where A is a subset of xyz_hist and B is a subset of xyz_oct 
    and the Z coordinate of point subset A is set to 1.
    """
    # 1. Pairwise squared distance, log ratio matrix
    epsilon = 1e-10
    d_gt = np.sum((xyz_oct[:, :, None] - xyz_oct[:, None, :])**2, axis=0)  # (n, n)
    d_est = np.sum((xyz_hist[:, :, None] - xyz_hist[:, None, :])**2, axis=0)  # (n, n)

    # Clamp very small distances 
    d_gt_safe = np.maximum(d_gt, epsilon)
    d_est_safe = np.maximum(d_est, epsilon)

    log_ratio_mat = 0.5 * np.log(d_est_safe / d_gt_safe)

    # 2. Score correspondence pairs
    min_costs = _score_correspondences(log_ratio_mat, thr1)
    sort_idx = np.argsort(min_costs)
    xyz_hist = xyz_hist[:, sort_idx]
    xyz_oct  = xyz_oct[:, sort_idx]

    # 3. pcr
    A, B = _core_PCR99a(xyz_oct, xyz_hist, log_ratio_mat, sort_idx, n_hypo, thr1, pcr99_inlier_thresh)

    # 4. plane fit ransac
    A, B = plane_ransac(A, B, n_iter, plane_inlier_thresh, z_dist_thresh,
                 penalty_threshold, xy_translation_penalty_weight)

    # 5. Final transform
    B_temp = B.copy()
    B_temp[2, :] = 1

    T = _compute_affine(A, B_temp)
    s,R,t = sRt_from_N_points(A,B_temp)
    return T, (s,R,t)
