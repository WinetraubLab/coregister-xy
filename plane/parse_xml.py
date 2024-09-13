import xml.etree.ElementTree as ET
import numpy as np
import transforms3d
from scipy.linalg import polar

class ParseXML:
    def __init__(self, source_points=None, dest_points=None, M=None):
        self.M = M # Transformation from source image to dest image coordinates
        self.source_points = source_points 
        self.dest_points = dest_points

    @classmethod
    def extract_data(cls, trakem_filepath, source_patch_num, dest_patch_num, landmarks_filepath=None, multi=False):
        """
        Function to get transform matrix, source and dest points from XML file of TrakEM2 project.
        Inputs:
            trakem_filepath: path to XML file.
            source_patch_num: the patch number assigned to your source image by ImageJ.
            dest_patch_num: patch number assigned to dest image by ImageJ.
            landmarks_filepath: the file with exported landmarks from both images.
        """
        tree = ET.parse(trakem_filepath)
        root = tree.getroot()
        # TODO error handling
        source_patch = root.find(f".//*[@oid='{source_patch_num}']")
        dest_patch = root.find(f".//*[@oid='{dest_patch_num}']")

        def extract_m(m_string):
            m_string = m_string.replace("matrix(", "").replace(")", "")
            m_string = list(map(float, m_string.split(',')))
            m = np.array([ [m_string[0], m_string[2], m_string[4]], [m_string[1], m_string[3], m_string[5]]  ])
            return m
        
        source_transform = source_patch.get('transform')
        if not source_transform:
                raise TypeError(f"No transform found for {source_patch}, patch id {source_patch_num}")
        source_transform = extract_m(source_transform)
        dest_transform = dest_patch.get('transform')
        if not dest_transform:
                raise TypeError(f"No transform found for {dest_patch}, patch id {dest_patch_num}")
        dest_transform = extract_m(dest_transform)

        source_transform = np.vstack([source_transform, [0.0, 0.0, 1.0]])
        dest_transform = np.vstack([dest_transform, [0.0, 0.0, 1.0]])

        # Get landmark points if file is provided
        source_points = []
        dest_points = []
        if landmarks_filepath:
            tree = ET.parse(landmarks_filepath)
            root = tree.getroot()
            source_points_list = root.findall(f".//*[@patch_id='{source_patch_num}']")
            dest_points_list = root.findall(f".//*[@patch_id='{dest_patch_num}']")
            for point in source_points_list:
                source_points.append([float(point.get('x')), float(point.get('y'))])
            for point in dest_points_list:
                dest_points.append([float(point.get('x')), float(point.get('y'))])
        source_points = np.array(source_points)
        dest_points = np.array(dest_points)

        if multi:
            return cls(
                M = dest_transform,
                source_points = dest_points,
                dest_points = source_points
            )
        else:
            return cls(
                M = source_transform,
                source_points = source_points,
                dest_points = dest_points
            )
    
    def compute_physical_params_old(self):
        """
        Compute physical representation of transform from matrix M.
        Returns:
            translation (x,y), rotation, scaling x, scaling y, shear.
        """
        M = self.M
        T, R, S, H = transforms3d.affines.decompose(M)
        tx, ty = T
        theta_rad = np.arctan2(R[1, 0], R[0, 0])
        theta_deg = np.degrees(theta_rad)
        sx, sy = S
        shear = H[0]

        return tx, ty, theta_deg, sx, sy, shear
         
    def compute_physical_params_svd(self, M):
        # SVD decomposition of a provided matrix
        A = np.array([[M[0,0], M[0,1]], [M[1,0], M[1,1]]])
        U, S, Vt = np.linalg.svd(A)
        print("ssssss:",S)
        s = np.sqrt(S[0] * S[1])  # TODO probably not always true
        # s = np.mean(S)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        theta_rad = np.arctan2(R[1,0], R[0,0])
        return s, theta_rad
    
    def compute_polar(self, M):
        """
        idk bro i did some math guessing and it worked eventually
        """
        # Separate out rotation
        R, S_shear = polar(M)
        print("Rotation matrix (R):")
        print(R)
        theta_rad = np.arctan2(R[1,0], R[0,0])
        print(np.degrees(theta_rad))

        # --------------------------------------------------------------------------

        # Use determinant to find scale
        sc = np.sqrt(np.linalg.det(S_shear))
        print("Scale from determinant: ",sc)
        S_inv = np.array([
            [1/sc,0,0],
            [0,1/sc,0],
            [0,0,1]
        ])
        # Isolate shear component
        H = S_shear @ S_inv
        print(H)

        # Find shear magnitude and direction 
        evals, evecs = np.linalg.eig(H)
        shear_magnitude = evals[0]
        shear_vector = evecs[0]
        if shear_magnitude < 0:
             shear_magnitude *= -1
             shear_vector *= -1
        shear_vector = shear_vector / np.linalg.norm(shear_vector)

        print("{:.2f}".format(shear_magnitude), shear_vector)

        return sc, np.degrees(theta_rad), shear_magnitude, shear_vector


        # --------------------------------------------------------------------------


        # M = S_shear
        # print("S_shear,",S_shear)
        # # Irreducible tensor decomp of remaining matrix for shear and scale
        # # Symmetric part
        # M_sym = 0.5 * (M + M.T)
        # # Antisymmetric (rotation) part
        # M_rotation = 0.5 * (M - M.T)
        # assert np.allclose(M_rotation, np.zeros(M_rotation.shape))

        # # Trace (expansion) part
        # trace = np.trace(M)
        # M_expansion = (trace / 2) * np.eye(3)
        # print("EXPANSION:", M_expansion[0,0])

        # # Symmetric trace-free (shear) part
        # M_shear = M_sym - M_expansion

        # evals, evecs = np.linalg.eig(M_shear)
        # shear_magnitude = evals[0]
        # shear_vector = evecs[0]
        # if shear_magnitude < 0:
        #      shear_magnitude *= -1
        #      shear_vector *= -1
        # shear_vector = shear_vector / np.linalg.norm(shear_vector)






        # # Compute trace of M
        # M = np.array([[S_shear[0,0], S_shear[0,1]], [S_shear[1,0], S_shear[1,1]]])
        # trace_M = np.trace(M)

        # # Compute the trace part (uniform scaling)
        # trace_part = (trace_M / 2) * np.eye(2)

        # # Compute the shear part
        # shear_part = M - trace_part

        # # Verify if the shear part is symmetric and trace-free
        # assert np.allclose(shear_part, shear_part.T)
        # trace_shear_part = np.trace(shear_part)

        # print("Trace Part:")
        # print(trace_part)
        # print("Shear Part:")
        # print(shear_part)
        # print("Trace of Shear Part (should be close to 0):", "{:.2f}".format(trace_shear_part))

        # evals, evecs = np.linalg.eig(shear_part)
        # shear_magnitude = evals[0]
        # shear_vector = evecs[0]
        # if shear_magnitude < 0:
        #      shear_magnitude *= -1
        #      shear_vector *= -1
        # shear_vector = shear_vector / np.linalg.norm(shear_vector)

        # print("{:.2f}".format(shear_magnitude), shear_vector)
    
    def compute_scale_rotation(self, M):
        M = np.array([[M[0,0], M[0,1]], [M[1,0], M[1,1]]])
        scale = np.sqrt(M[0,0] * M[1,1])
        s = scale * np.eye(2)
        s_inv = np.linalg.inv(s)
        R = M @ s_inv
        theta_rad = np.arctan2(R[0,0], R[1,0])
        return scale, theta_rad
    
    def compute_physical_svd_irreducible(self):
        """
        Compute physical representation of transform from matrix M.
        1) remove average uniform scaling and rotation components using SVD
        2) find shear from remaining matrix using irreducible tensor decomposition
        3) check that rotation and scaling are close to 0
        Returns:
            uniform scale, rotation, shear magnitude, shear vector
        """
        s, theta_rad = self.compute_physical_params_svd(self.M)

        S = 1/s * np.eye(2)
        R = np.array([
          [np.cos(theta_rad), -np.sin(theta_rad)],
          [np.sin(theta_rad), np.cos(theta_rad)]])
        print(self.M)
        A = np.array([[self.M[0,0], self.M[0,1]], [self.M[1,0], self.M[1,1]]])
        A = A @ S
        M = A @ np.linalg.inv(R)
        print(M)
        shear = M[0,0]

        print("Results:")
        print(s, np.degrees(theta_rad),shear)

        # b,c and a,d should be close
        # if not np.all(M == 0):
        #     assert M[0,0] * M[1,1] <= 0, "Cannot find irreducible matrix decomposition, a and d should be opposite signs"
        #     if M[1,0] != 0 and M[0,1] != 0:
        #         assert abs(abs(M[1,0]) - abs(M[0,1])) / ((abs(M[1,0]) + abs(M[0,1])) / 2) < 0.2, "Cannot find irreducible matrix decomposition, b and c not similar"
        #     if M[1,1] != 0 and M[0,0] != 0:
        #         assert abs(abs(M[0,0]) - abs(M[1,1])) / ((abs(M[0,0]) + abs(M[1,1])) / 2) < 0.2, "Cannot find irreducible matrix decomposition, a and d not similar"

        # Irreducible tensor decomp of remaining matrix for shear
        # Symmetric part
        M_sym = 0.5 * (M + M.T)
        # Antisymmetric (rotation) part
        M_rotation = 0.5 * (M - M.T)
        # Trace (expansion) part
        trace = np.trace(M)
        M_expansion = (trace / 2) * np.eye(2)

        # Symmetric trace-free (shear) part
        M_shear = M_sym - M_expansion

        evals, evecs = np.linalg.eig(M_shear)
        shear_magnitude = evals[0]
        shear_vector = evecs[0]
        if shear_magnitude < 0:
             shear_magnitude *= -1
             shear_vector *= -1
        shear_vector = shear_vector / np.linalg.norm(shear_vector)
        
        # scale and rotation should be close to 0
        # assert M_expansion[0,0] < 0.1
        # assert np.degrees(np.arctan2(M_rotation[0,0], M_rotation[1,0])) % 180 < 10

        # Reconstruct and remove shear from original M to find rotation and scaling.
        # TODO temporary?
        # outer_product = np.outer(shear_vector, shear_vector)
        # shear_matrix = np.eye(2) + shear_magnitude * outer_product
        # H_inv = np.linalg.inv(shear_matrix)
        # A_no_shear = A @ H_inv
        # s2, theta2 = self.compute_physical_params_svd(A_no_shear)
        
        return s, np.degrees(theta_rad), shear_magnitude, shear_vector, self.M[0,2], self.M[1,2]
    
    def transform_points(self, points):
         points = np.array(points)
         transformed_points = self.M @ points
         return transformed_points

    def find_transformation_error_from_points(self, source_points, dest_points):
        """
        This function calculates the transformation error between a set of source and destination 
        points. Inputs can be points other than the landmarks already stored in the class object.
        Inputs:
            source_points: an array or list of points (x,y).
            dest_points: must have same shape as source_points.
        """
        source_points = np.array(source_points)
        dest_points = np.array(dest_points)
        assert source_points.shape == dest_points.shape

        if source_points.shape[-1] == 2:
             ones_column = np.ones((source_points.shape[0], 1))
             source_points = np.hstack((source_points, ones_column))
             dest_points = np.hstack((dest_points, ones_column))

        transformed_points = source_points @ self.M.T
        assert transformed_points.shape == source_points.shape

        transformed_points = transformed_points[:, :-1]
        source_points = source_points[:, :-1]
        dest_points = dest_points[:, :-1]

        distances = np.linalg.norm(transformed_points-dest_points, axis=1)
        avg_err = np.mean(distances)
        return avg_err

    def find_transformation_error(self):
        """
        This function calculates the transformation error between the source and dest points
        from LANDMARKS.XML. Must be initial landmarks selected BEFORE any alignment was done.
        Returns:
            Average distance between respective landmarks after alignment
        """
        return self.find_transformation_error_from_points(self.source_points, self.dest_points)
    
    def calc_real_scale(self, original_um_per_pixel=1, angle=0):
        """
        Function calculates the distance represented per pixel after transformation, along
        a given direction (useful for calculating the distance between 2 barcodes).
        Returns: distance represented per pixel along given angle (default 0 = along x axis).
        Angle increases counterclockwise starting at x axis.
        """
        s, theta_deg, shear_magnitude, shear_vector, tx, ty = self.compute_physical_svd_irreducible()
        # rel_angle = angle # - theta_deg
        rel_angle = theta_deg
        rads = np.radians(rel_angle)

        shear_vector = shear_vector/np.linalg.norm(shear_vector)
        outer_product = np.outer(shear_vector, shear_vector)
        H = np.eye(2) + shear_magnitude * outer_product
        det = np.linalg.det(H)
        H = H / np.sqrt(det)

        S = s * np.eye(2)
        R = np.array([[np.cos(rads), -np.sin(rads)],
                    [np.sin(rads), np.cos(rads)]])
        A = R @ S @ H
        v = np.array([np.cos(rads), np.sin(rads)])
        vt = A @ v
        scaling = np.linalg.norm(vt)
        print("scaling:",scaling)

        dist_per_pixel = original_um_per_pixel / scaling
        return dist_per_pixel
    