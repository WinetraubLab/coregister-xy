import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

class FitPlaneTPS:

    def __init__(self, W=None, control_points=None):
        self.W = W
        self.control_points=control_points

    @classmethod
    def from_template_centers(
        cls, template_center_positions_uv_pix, template_center_positions_xyz_mm, print_inputs = False):

        template_center_positions_uv_pix = np.array(template_center_positions_uv_pix)
        template_center_positions_xyz_mm = np.array(template_center_positions_xyz_mm)

        # Print inputs
        if print_inputs:
            txt = ("FitPlane.from_template_centers(" +
                   json.dumps(template_center_positions_uv_pix.tolist()) + "," +
                   json.dumps(template_center_positions_xyz_mm.tolist()))   
            txt += ')'
            print(txt)

        num_points = template_center_positions_uv_pix.shape[0]

        # TPS kernel K: compute euclidean distance between all pairs of control points
        # K defines how much each control point contributes to the deformation
        dist_matrix = np.linalg.norm(template_center_positions_uv_pix[:, None, :] - template_center_positions_uv_pix[None, :, :], axis=2) ** 2  # Broadcast (N,d) to (N,1,d) and (1,N,d)
        K = dist_matrix * np.log(dist_matrix + 1e-10)  # U(r) = r^2 log r (r = dist)

        # System matrix
        # P = affine part of the transformation.
        P = np.hstack((np.ones((num_points, 1)), template_center_positions_uv_pix))
        L = np.block([
            [K, P],
            [P.T, np.zeros((3, 3))]
        ])

        # Solve for TPS weights
        # Y = np.vstack([template_center_positions_xyz_mm, np.zeros((3, 2))]) # RHS of equation
        Y = np.vstack([template_center_positions_xyz_mm, np.zeros((3, 3))])  # Change (3,2) -> (3,3)

        W = np.linalg.solve(L, Y) # weights that will be used to warp new points

        fp = cls(W=W, control_points=template_center_positions_uv_pix)
        return fp
    