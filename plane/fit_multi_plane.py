import numpy as np
from plane.parse_xml import ParseXML

class FitMultiPlane:
    def __init__(self, landmarks, real_centers):
        """
        - list of barcodes contained in object
            - contains self.scale, self.theta_deg, self.shear_magnitude, self.shear_vector, self.tx, self.ty, self.z
            - Real positions of each barcode center as defined by photobleach script
        - Adjacency matrix to store distances between pairs of barcodes.
        - ??? to store angles between 3 barcodes for xy, xz, yz planes 
        """
        self.landmarks = landmarks
        self.real_centers = real_centers
        self.distances = self.calc_distances()
        self.angles = self.calc_angles()


    @classmethod
    def from_aligned_landmarks(cls, landmarks, real_centers):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 
            - Individual barcodes: xy center position, rotation angle on xy plane, shear, scaling
            - Distances and angles between barcodes

        Inputs:
            - landmarks: array-like of fitted planes (ParseXML objects)
            - real_centers: array-like with the locations of the landmarks as defined by photobleach script
        """
        return cls(landmarks, real_centers)

    def calc_distances(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        """
        n = len(self.landmarks)
        adj = np.zeros((n,n))
        for i in range(0, n):
            for j in range(0,n):
                dist = np.sqrt((self.landmarks[i].tx - self.landmarks[j].tx)**2 + (self.landmarks[i].ty - self.landmarks[j].ty)**2)
                adj[i,j] = dist
        return adj


    def calc_angles(self):
        pass
    
    def calc_angles_xy(self):
        pass

    def calc_angles_xz(self):
        pass

    def calc_angles_yz(self):
        pass

    def fit_best_plane(self):
        """
        Return a single best fit plane for all given barcodes
        """
        pass