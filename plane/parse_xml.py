import xml.etree.ElementTree as ET
import numpy as np
import transforms3d

class ParseXML:
    def __init__(self, source_points=None, dest_points=None, M=None):
        self.M = M # Transformation from source image to dest image coordinates
        self.source_points = source_points 
        self.dest_points = dest_points

    @classmethod
    def extract_data(cls, trakem_filepath, source_patch_num, dest_patch_num, landmarks_filepath=None):
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

        # dest_transform is supposed to be identity matrix, so if not, add the inverse transform to source_transform
        eye = np.eye(2,3)
        if not np.allclose(eye, dest_transform):
            source_transform = np.linalg.inv(dest_transform) @ source_transform
        source_transform = np.vstack([source_transform, [0.0, 0.0, 1.0]])

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

        return cls(
            M = source_transform,
            source_points = source_points,
            dest_points = dest_points
        )
    
    def compute_physical_params(self):
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

        transformed_points = self.M @ self.dest_points 
        assert transformed_points.shape == self.source_points.shape

        distances = np.linalg.norm(transformed_points-self.source_points, axis=1)
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