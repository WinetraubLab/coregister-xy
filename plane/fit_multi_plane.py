
class FitMultiPlane:
    def __init__(self):
        """
        - list of barcodes contained in object
            - contains self.scale, self.theta_deg, self.shear_magnitude, self.shear_vector, self.tx, self.ty, self.z
        - Adjacency matrix to store distances between pairs of barcodes.
        - ??? to store angles between 3 barcodes for xy, xz, yz planes 
        """
        pass

    @classmethod
    def from_imagej_xml(cls):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 
            - Individual barcodes: xy center position, rotation angle on xy plane, shear, scaling
            - Distances and angles between barcodes
        """
        pass

    def calc_distances(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        """
        pass
    
    def calc_angles_xy(self):
        pass

    def calc_angles_xz(self):
        pass

    def calc_angles_yz(self):
        pass