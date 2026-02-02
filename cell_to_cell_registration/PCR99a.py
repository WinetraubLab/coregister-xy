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

def _score_correspondences(log_ratio_mat, thr1, bin_width=0.1):
        """
        Step 2 of PCR99a.
        Inputs:
            log_ratio_mat: (n, n) matrix
            thr1: clipping threshold

        Returns:
            min_costs: (n,) array of minimum costs per row
        """
        n = log_ratio_mat.shape[0]
        min_costs = np.empty(n, dtype=np.float32)

        for i in range(n):
            lr = log_ratio_mat[i]
            
            # Remove nans
            lr = lr[~np.isnan(lr)]
            if lr.size == 0:
                min_costs[i] = np.inf
                continue
            
            min_lr, max_lr = lr.min(), lr.max()
            if max_lr == min_lr:
                min_costs[i] = 0.0
                continue
            
            # Bin centers
            num_bins = max(1, round((max_lr - min_lr) / bin_width))
            centers = np.linspace(min_lr, max_lr, num_bins)

            # r shape: (num_centers, len(lr))
            r = np.abs(lr[np.newaxis, :] - centers[:, np.newaxis])
            np.clip(r, None, thr1, out=r)  # in-place clipping

            # Sum over each center row
            costs = np.sum(r, axis=1)
            min_costs[i] = costs.min()
        
        return min_costs

def _enforce_one_to_one_in_inliers(xyz_gt, xyz_est, candidate_indices, errors, coord_tolerance=1e-6):
    """
    Helper function to enforce one-to-one matching when selecting inliers.
    Given candidate inlier indices and their errors, returns a subset that ensures
    each point appears at most once.
    """
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)
    
    # Round coordinates for matching
    xyz_gt_rounded = np.round(xyz_gt / coord_tolerance) * coord_tolerance
    xyz_est_rounded = np.round(xyz_est / coord_tolerance) * coord_tolerance
    
    # Sort by error (best first)
    sorted_idx = np.argsort(errors[candidate_indices])
    sorted_candidates = candidate_indices[sorted_idx]
    
    selected = []
    gt_used = set()
    est_used = set()
    
    for idx in sorted_candidates:
        gt_key = tuple(xyz_gt_rounded[:, idx])
        est_key = tuple(xyz_est_rounded[:, idx])
        
        if gt_key not in gt_used and est_key not in est_used:
            selected.append(idx)
            gt_used.add(gt_key)
            est_used.add(est_key)
    
    return np.array(selected, dtype=int)

def _core_PCR99a(xyz_gt, xyz_est, log_ratio_mat, sort_idx, n_hypo, thr1, pcr99_inlier_thresh, 
                 enforce_one_to_one=True, coord_tolerance=1e-6):
    """
    This function contains the Python adaptation of the original PCR99a algorithm.
    Robustly estimate correspondences between two 3D point sets by generating hypotheses 
    from point triplets.
    Inputs:
        xyz_gt: array (3,n) of reference points.
        xyz_est: array (3,n) of points to match to reference points.
        log_ratio_mat: (n,n) precomputed log-ratio matrix between points, used for prescreening triplets.
        sort_idx: array (n,) Indices giving the sorted order of points (used to map triplets to original indices).
        n_hypo: Number of hypotheses to batch together before evaluating inliers. Smaller values are recommended for small point sets.
        thr1: Log-ratio consistency threshold for prescreening candidate triplets.
        pcr99_inlier_thresh: Distance threshold for final inliers.
        enforce_one_to_one: If True, enforce one-to-one matching constraint during inlier selection.
        coord_tolerance: Tolerance for considering two points as the same.
    Returns:
        A, B: arrays of corresponding inlier point pairs.
    """
    n = xyz_gt.shape[1]

    # Configurable parameters
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
    total_iters = 0
    break_loop = False

    for s in range(6, n + (n - 1) + (n - 2)):
        i_min = max(1, s - n - (n - 1))
        i_max = (s - 3) // 3
        if break_loop:
            break

        for i in range(i_min, i_max + 1):
            j_min = max(i + 1, s - i - n)
            j_max = (s - i - 1) // 2
            if break_loop:
                break

            for j in range(j_min, j_max + 1):
                k = s - i - j
                if not (0 <= k < n):
                    continue
                if break_loop:
                    break

                # Map to original indices
                i_old = sort_idx[i]
                j_old = sort_idx[j]
                k_old = sort_idx[k]

                # Prescreening
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
                    total_iters += 1

                    Ex = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[0, :].reshape(-1, 1)]) @ Sx_mat
                    Ey = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[1, :].reshape(-1, 1)]) @ Sy_mat
                    Ez = np.hstack([-xyz_gt.T, -np.ones((n, 1)), xyz_est[2, :].reshape(-1, 1)]) @ Sz_mat

                    E = Ex**2 + Ey**2 + Ez**2

                    nInliers = np.sum(E <= e_thr, axis=0)
                    max_nInliers = np.max(nInliers)
                    idx = np.argmax(nInliers)

                    # Check if best so far
                    candidate_inliers = np.where(E[:, idx] <= e_thr)[0]
                    
                    # Enforce one-to-one matching if requested
                    if enforce_one_to_one and len(candidate_inliers) > 0:
                        candidate_inliers = _enforce_one_to_one_in_inliers(
                            xyz_gt, xyz_est, candidate_inliers, E[:, idx], coord_tolerance
                        )
                        max_nInliers = len(candidate_inliers)
                    
                    if max_nInliers > max_max_nInliers:
                        max_max_nInliers = max_nInliers
                        idx_inliers = candidate_inliers
                        no_improvement_iters = 0
                    else:
                        no_improvement_iters += 1

                    if early_stop_on and \
                      max_max_nInliers >= min_inliers_to_trigger_stop and \
                      no_improvement_iters >= no_improvement_limit:
                        break_loop = True
                        break  # from inner loop

                    # after 1000 iterations, allow looser exit
                    if total_iters > 1000:
                        if max_max_nInliers >= max(9, round(n * 0.009)):
                            break_loop = True
                            break

    if len(idx_inliers) == 0:
        print("No inliers found.")
        return np.array([]), np.array([])

    A = xyz_gt[:, idx_inliers]
    B = xyz_est[:, idx_inliers]

    return A, B

def plane_ransac(points_from_oct, points_from_hist, n_iter = 2000, 
                 plane_inlier_thresh=5, y_dist_thresh=4,
                 penalty_threshold=8, xz_translation_penalty_weight=1,
                 enforce_one_to_one=True, coord_tolerance=1e-6, transformation_residual_thresh=None, use_xz_distance=False):
    """
    RANSAC round 2: plane-based inlier set refinement.
    Params:
        points_from_oct, points_from_hist: corresponding point pairs from initial PCR99 inlier selection.
        n_iter: RANSAC iterations to perform.
        plane_inlier_thresh: The maximum perpendicular distance from a candidate plane at which a point is 
            still considered an inlier during RANSAC iterations.
        y_dist_thresh: threshold on the distance of points to the final plane (for planes close to parallel with y-axis). 
            It defines which points are retained as the final inlier set.
        penalty_threshold: amount of XZ translation between the two point sets that is acceptable 
            before incurring a score penalty.
        xz_translation_penalty_weight: scaling factor for how severely to penalize XZ translation beyond penalty_threshold.
        enforce_one_to_one: If True, enforce one-to-one matching constraint during inlier selection.
        coord_tolerance: Tolerance for considering two points as the same.
        transformation_residual_thresh: Optional threshold for transformation residuals. Pairs with residuals
            above this threshold are filtered out. Lower = stricter geometric consistency.
        use_xz_distance: If True, use XZ distance for transformation residual computation.
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

        # Penalize large xz translations
        A_sub = points_from_oct[:, inliers_candidate]
        B_sub = points_from_hist[:, inliers_candidate]

        if A_sub.shape[1] < 1:
            continue

        s_temp, R_temp, t_temp = sRt_from_N_points(A_sub, B_sub)

        xz_trans = np.linalg.norm([t_temp[0], t_temp[2]])

        score = num_inliers_candidate - xz_translation_penalty_weight * max(0, xz_trans - penalty_threshold)

        # Update best if score improved
        if score > best_score:
            best_score = score
            best_plane_normal = plane_normal_candidate
            best_plane_point = mean_subset

    best_plane_normal = best_plane_normal / np.linalg.norm(best_plane_normal)

    # Distance to the plane (signed projection) - for planes close to parallel with y-axis
    vecs_to_plane = points_from_oct - best_plane_point.reshape(-1, 1)  # shape (3×N)
    y_dists = np.abs(best_plane_normal.T @ vecs_to_plane).flatten()  # shape (N,)
    valid_mask = y_dists <= y_dist_thresh

    candidate_inliers = np.where(valid_mask)[0]
    
    # Compute transformation residuals for ranking and filtering
    ranking_errors = y_dists.copy()  # Default to plane distance for ranking
    if len(candidate_inliers) > 0:
        A_candidate = points_from_oct[:, candidate_inliers]
        B_candidate = points_from_hist[:, candidate_inliers]
        B_temp = B_candidate.copy()
        B_temp[1, :] = 1
        
        try:
            T = _compute_affine(A_candidate, B_temp)
            A_homogeneous = np.vstack([A_candidate, np.ones((1, A_candidate.shape[1]))])
            A_transformed = (T @ A_homogeneous)[:3, :]
            
            if use_xz_distance:
                residuals = np.sqrt((A_transformed[0, :] - B_candidate[0, :])**2 + 
                                   (A_transformed[2, :] - B_candidate[2, :])**2)
            else:
                residuals = np.sqrt(np.sum((A_transformed - B_candidate)**2, axis=0))
            
            # Use transformation residuals for ranking (better metric)
            ranking_errors_full = np.full(points_from_oct.shape[1], np.inf)
            ranking_errors_full[candidate_inliers] = residuals
            ranking_errors = ranking_errors_full
            
            # Apply transformation residual threshold if specified
            if transformation_residual_thresh is not None:
                residual_mask = residuals <= transformation_residual_thresh
                candidate_inliers = candidate_inliers[residual_mask]
        except:
            pass  # If transformation fails, use plane distance
    
    # Enforce one-to-one matching if requested
    # if enforce_one_to_one and len(candidate_inliers) > 0:
    #     candidate_inliers = _enforce_one_to_one_in_inliers(
    #         points_from_oct, points_from_hist, candidate_inliers, 
    #         ranking_errors, coord_tolerance
    #     )
    
    A_final = points_from_oct[:, candidate_inliers]
    B_final = points_from_hist[:, candidate_inliers]

    return A_final, B_final

def enforce_one_to_one_matching(A, B, coord_tolerance=1e-6, use_xz_distance=False, pair_scores=None, 
                                 use_transformation_consistency=True, transformation_residual_thresh=None):
    """
    Enforce one-to-one matching constraint: each OCT point matches at most one hist point,
    and each hist point matches at most one OCT point.
    
    Uses transformation consistency to determine which pairs are correct: computes a consensus
    transformation and selects pairs that best fit this transformation.
    
    Inputs:
        A: (3, n) array of OCT points
        B: (3, n) array of hist points (corresponding pairs)
        coord_tolerance: tolerance for considering two points as the same (for coordinate matching)
        use_xz_distance: If True, use XZ distance for residual computation; otherwise use 3D distance
        pair_scores: Optional (n,) array of scores for each pair. Lower scores = better matches.
            If provided, will use these scores instead of transformation consistency.
        use_transformation_consistency: If True, use transformation residual to determine best matches.
            This evaluates how well each pair fits the overall transformation consensus.
        transformation_residual_thresh: Optional threshold for transformation residuals. Pairs with
            residuals above this threshold will be filtered out. Lower = stricter geometric consistency.
            If None, no filtering is applied (only ranking is used).
    
    Returns:
        A_unique, B_unique: (3, m) arrays with one-to-one matched pairs, m <= n
    """
    if A.shape[1] == 0:
        return A, B
    
    n = A.shape[1]
    
    # Round coordinates for matching (using tolerance)
    A_rounded = np.round(A / coord_tolerance) * coord_tolerance
    B_rounded = np.round(B / coord_tolerance) * coord_tolerance
    
    # Group pairs by unique points
    oct_groups = {}  # Maps OCT point -> list of (pair_index, hist_point)
    hist_groups = {}  # Maps hist point -> list of (pair_index, oct_point)
    
    for i in range(n):
        oct_key = tuple(A_rounded[:, i])
        hist_key = tuple(B_rounded[:, i])
        
        if oct_key not in oct_groups:
            oct_groups[oct_key] = []
        oct_groups[oct_key].append((i, hist_key))
        
        if hist_key not in hist_groups:
            hist_groups[hist_key] = []
        hist_groups[hist_key].append((i, oct_key))
    
    # Compute scores for each pair
    if pair_scores is not None:
        # Use provided scores
        scores = pair_scores
    elif use_transformation_consistency:
        # Compute transformation from all pairs and use residuals as scores
        B_temp = B.copy()
        B_temp[1, :] = 1  # Set Y coordinate to 1 for transformation
        
        try:
            T = _compute_affine(A, B_temp)
            # Transform A points
            A_homogeneous = np.vstack([A, np.ones((1, A.shape[1]))])
            A_transformed = (T @ A_homogeneous)[:3, :]
            
            # Compute residuals
            if use_xz_distance:
                # XZ in-plane residual
                scores = np.sqrt((A_transformed[0, :] - B[0, :])**2 + (A_transformed[2, :] - B[2, :])**2)
            else:
                # Full 3D residual
                scores = np.sqrt(np.sum((A_transformed - B)**2, axis=0))
        except:
            # Fallback to distance if transformation fails
            if use_xz_distance:
                scores = np.sqrt((A[0, :] - B[0, :])**2 + (A[2, :] - B[2, :])**2)
            else:
                scores = np.sqrt(np.sum((A - B)**2, axis=0))
    else:
        # Use simple distance
        if use_xz_distance:
            scores = np.sqrt((A[0, :] - B[0, :])**2 + (A[2, :] - B[2, :])**2)
        else:
            scores = np.sqrt(np.sum((A - B)**2, axis=0))
    
    # Filter pairs by residual threshold if specified (for geometric consistency)
    if transformation_residual_thresh is not None:
        valid_mask = scores <= transformation_residual_thresh
        if not np.any(valid_mask):
            # If no pairs pass threshold, return empty
            return np.array([]).reshape(3, 0), np.array([]).reshape(3, 0)
        # Only consider pairs that pass the threshold
        candidate_indices = np.where(valid_mask)[0]
        candidate_scores = scores[candidate_indices]
        pair_indices_sorted = candidate_indices[np.argsort(candidate_scores)]
    else:
        # Sort all pairs by score (best first)
        pair_indices_sorted = np.argsort(scores)
    
    # Greedy selection: for each unique point, keep the pair with best score
    # that doesn't conflict with already selected pairs
    selected_pairs = set()
    oct_used = set()
    hist_used = set()
    
    for i in pair_indices_sorted:
        oct_key = tuple(A_rounded[:, i])
        hist_key = tuple(B_rounded[:, i])
        
        # Skip if either point is already matched
        if oct_key in oct_used or hist_key in hist_used:
            continue
        
        # This pair is valid - add it
        selected_pairs.add(i)
        oct_used.add(oct_key)
        hist_used.add(hist_key)
    
    if len(selected_pairs) == 0:
        return np.array([]).reshape(3, 0), np.array([]).reshape(3, 0)
    
    valid_indices = sorted(selected_pairs)
    A_unique = A[:, valid_indices]
    B_unique = B[:, valid_indices]
    
    return A_unique, B_unique

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

def calculate_affine_alignment(xyz_oct, xyz_hist, n_hypo=1000, thr1=0.03, pcr99_inlier_thresh=50, n_iter=2000, plane_inlier_thresh=5, y_dist_thresh=4,
                 penalty_threshold=8, xz_translation_penalty_weight=1, enforce_one_to_one=True, coord_tolerance=1e-6, use_xz_distance=False,
                 transformation_residual_thresh=None):
    """
    Run full alignment algorithm. 
    Inputs:
        xyz_oct: Reference point set from 3D OCT segmentations, in pixels. (3,n)
        xyz_hist: Candidate set to align, from 2D histology image, in pixels. (3,n)
        n_hypo: Number of hypotheses to batch together before evaluating inliers. Smaller values are recommended for small point sets.
        thr1: Log-ratio consistency threshold for prescreening candidate triplets.
        pcr99_inlier_thresh: threshold for considering points inliers in PCR99a (squared distance). 
            LOWER = STRICTER geometric consistency in initial matching.
        n_iter: RANSAC iterations to perform.
        plane_inlier_thresh: The maximum perpendicular distance from a candidate plane at which a point is 
            still considered an inlier during RANSAC iterations. LOWER = STRICTER geometric consistency in plane fitting.
        y_dist_thresh: threshold on the distance of points to the final plane (for planes close to parallel with y-axis). 
            It defines which points are retained as the final inlier set. LOWER = STRICTER geometric consistency.
        penalty_threshold: amount of XZ translation between the two point sets that is acceptable 
            before incurring a score penalty.
        xz_translation_penalty_weight: scaling factor for how severely to penalize XZ translation beyond penalty_threshold.
        enforce_one_to_one: If True, enforce one-to-one matching constraint during PCR99a and plane RANSAC
            inlier selection (each point matches at most one point). This is integrated into the algorithms
            themselves, not as post-processing.
        coord_tolerance: Tolerance for considering two points as the same when enforcing one-to-one matching.
        use_xz_distance: If True, use XZ distance for transformation residual computation; otherwise use 3D distance.
        transformation_residual_thresh: Threshold for transformation residuals in plane RANSAC.
            Pairs with residuals above this threshold are filtered out. LOWER = STRICTER geometric consistency.
            If None, no filtering is applied. Recommended: 5-20 for tight consistency.
    
    Returns:
    T: transformation matrix such that T @ A = B, where A is a subset of xyz_hist and B is a subset of xyz_oct 
    and the Y coordinate of point subset A is set to 1.
    s, R, t: T separated into scale, rotation, and translation components.
    A : filtered inliers from OCT point set.
    B: filtered inliers from histology point set corresponding to A.
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

    # 3. pcr (with one-to-one matching built in)
    A, B = _core_PCR99a(xyz_oct, xyz_hist, log_ratio_mat, sort_idx, n_hypo, thr1, pcr99_inlier_thresh,
                        enforce_one_to_one=enforce_one_to_one, coord_tolerance=coord_tolerance)

    # 4. plane fit ransac (with one-to-one matching and transformation consistency built in)
    A, B = plane_ransac(A, B, n_iter, plane_inlier_thresh, y_dist_thresh,
                 penalty_threshold, xz_translation_penalty_weight,
                 enforce_one_to_one=enforce_one_to_one, coord_tolerance=coord_tolerance,
                 transformation_residual_thresh=transformation_residual_thresh, use_xz_distance=use_xz_distance)

    # 5. Final transform
    if A.shape[1] == 0:
        print("No valid pairs after one-to-one matching.")
        return None, None, np.array([]).reshape(3, 0), np.array([]).reshape(3, 0)
    
    B_temp = B.copy()
    B_temp[1, :] = 1

    T = _compute_affine(A, B_temp)
    s,R,t = sRt_from_N_points(A,B_temp)
    return T, (s,R,t), A, B
