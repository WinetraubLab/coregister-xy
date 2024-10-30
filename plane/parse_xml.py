import xml.etree.ElementTree as ET
import numpy as np
from scipy.linalg import polar

class ParseXML:
    def __init__(self, source_points=None, dest_points=None, M=None, z=0, template_id = None):
        self.M = M # Transformation from source image to dest image coordinates
        self.source_points = source_points 
        self.dest_points = dest_points
        self.z = z
        self.scale, self.theta_deg, self.shear_magnitude, self.shear_vector, self.tx, self.ty = self._compute_physical(self.M)
        self.template_id = template_id

    @classmethod
    def from_imagej_xml(cls, trakem_filepath, source_patch_number, dest_patch_number, z=0, landmarks_filepath=None, multi=False):
        """
        Function to get transform matrix, source and dest points from XML file of TrakEM2 project.
        Inputs:
            trakem_filepath: path to XML file.
            source_patch_number: the patch number assigned to your source image by ImageJ.
            dest_patch_number: patch number assigned to dest image by ImageJ.
            landmarks_filepath: the file with exported landmarks from both images.
        Outputs: returns a ParseXML object describing one pair of photobleach barcodes.
        """
        tree = ET.parse(trakem_filepath)
        root = tree.getroot()
        source_patch = root.find(f".//*[@oid='{source_patch_number}']")
        dest_patch = root.find(f".//*[@oid='{dest_patch_number}']")

        def extract_m(m_string):
            m_string = m_string.replace("matrix(", "").replace(")", "")
            m_string = list(map(float, m_string.split(',')))
            m = np.array([ [m_string[0], m_string[2], m_string[4]], [m_string[1], m_string[3], m_string[5]]  ])
            return m
        
        source_transform = source_patch.get('transform')
        if not source_transform:
                raise TypeError(f"No transform found for {source_patch}, patch id {source_patch_number}")
        source_transform = extract_m(source_transform)
        dest_transform = dest_patch.get('transform')
        if not dest_transform:
                raise TypeError(f"No transform found for {dest_patch}, patch id {dest_patch_number}")
        dest_transform = extract_m(dest_transform)

        source_transform = np.vstack([source_transform, [0.0, 0.0, 1.0]])
        dest_transform = np.vstack([dest_transform, [0.0, 0.0, 1.0]])

        # Get landmark points if file is provided
        source_points = []
        dest_points = []
        if landmarks_filepath:
            tree = ET.parse(landmarks_filepath)
            root = tree.getroot()
            source_points_list = root.findall(f".//*[@patch_id='{source_patch_number}']")
            dest_points_list = root.findall(f".//*[@patch_id='{dest_patch_number}']")
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
                dest_points = source_points,
                z=z,
                template_id = dest_patch_number
            )
        else:
            return cls(
                M = source_transform,
                source_points = source_points,
                dest_points = dest_points,
                z=z,
                template_id = dest_patch_number
            )
    
    def set_M(self, M):
        """Set or update M."""
        self.M = M
        self.scale, self.theta_deg, self.shear_magnitude, self.shear_vector, self.tx, self.ty = self._compute_physical(self.M)

    def _compute_physical(self, M):
        """
        Inputs: 3x3 affine transformation matrix
        Returns: uniform scaling, rotation angle (deg), volume-preserving shear magnitude and vector, translation x, translation y
        Here, a shear magnitude of 0 represents no shear. Expansion in the direction if > 0, squashing it if < 0
        """
        # Separate out rotation
        M_ul = np.array([[M[0,0], M[0,1]], [M[1,0], M[1,1]]])
        R, S_shear = polar(M_ul)
        theta_rad = np.arctan2(R[1,0], R[0,0])

        # Use determinant to find scale
        sc = np.sqrt(np.linalg.det(S_shear))
        S_inv = np.array([
            [1/sc,0],
            [0,1/sc],
        ])
        # Isolate shear component
        H = S_shear @ S_inv

        # Find shear magnitude and direction 
        evals, evecs = np.linalg.eig(H)
        shear_magnitude = evals[0]
        shear_vector = evecs[0]
        if shear_magnitude < 0:
             shear_magnitude *= -1
             shear_vector *= -1
        shear_vector = shear_vector / np.linalg.norm(shear_vector)

        return sc, np.degrees(theta_rad), shear_magnitude-1, shear_vector, M[0,2], M[1,2]
    
    def transform_points(self, points):
        """
        Inputs: array or list of points (x,y,z)
        Outputs: np array of transformed points
        """
        points = np.array(points)
        transformed_points = self.M @ points
        return transformed_points

    def find_transformation_error(self, source_points=None, dest_points=None):
        """
        This function calculates the transformation error between a set of source and destination 
        points. Inputs can be points other than the landmarks already stored in the class object.
        Inputs:
            source_points: an array or list of points (x,y).
            dest_points: must have same shape as source_points.
        Outputs: average error between transformed source_points and dest_points.
        """
        if not source_points:
             source_points = self.source_points
             dest_points = self.dest_points
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
    