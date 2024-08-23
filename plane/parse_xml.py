import xml.etree.ElementTree as ET
import numpy as np

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
            filepath: path to XML file
        """
        tree = ET.parse(trakem_filepath)
        root = tree.getroot()
        # TODO error handling
        source_patch = root.findall(f".//*[@oid='{source_patch_num}']")
        dest_patch = root.findall(f".//*[@oid='{dest_patch_num}']")
        print(source_patch.tag, dest_patch.tag)

        def extract_m(m_string):
            m_string = m_string.replace("matrix(", "").replace(")", "")
            m_string = list(map(float, m_string.split(',')))
            m = np.array([ [m_string[0], m_string[2], m_string[4]], [m_string[1], m_string[3], m_string[5]]  ])
            return m
        
        source_transform = source_patch.attrib['transform']
        if not source_transform:
                raise TypeError(f"No transform found for {source_patch}, patch id {source_patch_num}")
        source_transform = extract_m(source_transform)
        dest_transform = dest_patch.attrib['transform']
        if not dest_transform:
                raise TypeError(f"No transform found for {dest_patch}, patch id {dest_patch_num}")
        dest_transform = extract_m(dest_transform)

        # dest_transform is supposed to be identity matrix, so if not, add the inverse transform to source_transform
        eye = np.array([
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]
        ])
        if not np.allclose(eye, dest_transform):
            source_transform = np.linalg.inv(dest_transform) @ source_transform

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
