def calculate_affine_alignment(xyz_oct, xyz_hist, n_hypo=1000, thr1=0.03, sigma=2, thr2=5):
    """
    Run full alignment algorithm. 
    Inputs:
        xyz_oct: Reference point set from 3D OCT segmentations, in pixels. (3,n)
        xyz_hist: Candidate set to align, from 2D histology image, in pixels. (3,n)
        n_hypo: Number of hypotheses to batch together before evaluating inliers. Smaller values are recommended for small point sets.
        thr1: Log-ratio consistency threshold for prescreening candidate triplets.
        sigma: Noise scaling factor used in the inlier distance threshold.
        thr2: Distance threshold for final inliers, in pixels.
    Returns:
    T: transformation matrix such that T @ A = B, where A is a subset of xyz_hist and B is a subset of xyz_oct 
    and the Z coordinate of point subset A is set to 1.
    """
    # 1. Pairwise squared distance, log ratio matrix
    d_gt = np.sum((xyz_oct[:, :, None] - xyz_oct[:, None, :])**2, axis=0)   # (n, n)
    d_est = np.sum((xyz_hist[:, :, None] - xyz_hist[:, None, :])**2, axis=0) # (n, n)
    log_ratio_mat = 0.5 * np.log(d_est / d_gt)

    # 2. Score correspondence pairs
    min_costs = _score_correspondences(log_ratio_mat, thr1)
    sort_idx = np.argsort(min_costs)
    xyz_hist = xyz_hist[:, sort_idx]
    xyz_oct  = xyz_oct[:, sort_idx]

    # 3. pcr
    A, B = _core_PCR99a(xyz_oct, xyz_hist, log_ratio_mat, sort_idx, n_hypo, thr1, sigma, thr2)

    # 4. plane fit ransac
    A, B = plane_ransac(A, B)

    # 5. Final transform
    B_temp = B.copy()
    B_temp[2, :] = 1

    T = compute_affine(A, B_temp)
    return T
