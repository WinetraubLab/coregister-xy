import json
import numpy as np

class FitPlane:
    
    """ Begin constractor methods """
    def __init__(self, M=None):
        self.M = M
 
    
    @classmethod
    def from_fitting_points_between_fluorescence_image_and_template(cls, 
        source_image_points, dest_image_points):
        """
        This function initialize FitPlane by points on both source and destination images.
        
        INPUTS:
            source_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
            dest_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
        """
        # Type conversion

        source_image_points = np.array(source_image_points, dtype=np.float32)
        dest_image_points = np.array(dest_image_points, dtype=np.float32)
        assert source_image_points.shape == dest_image_points.shape, "Number of points must match"

        # Create least squared matrix equations for quadratic transformation
        A = []
        B = []
        for i in range(6):
            x_dest, y_dest = dest_image_points[i]
            x, y = source_image_points[i]
            A.append([x**2,y**2,x*y,x,y,1])
            B.append([x_dest, y_dest])
        B = np.array(B)
        A = np.array(A, dtype=np.float32)
        assert B.shape == (6,2), "Shape of matrix B incorrect"
        assert A.shape == (6,6), "Shape of matrix A incorrect"

        # Solve least squared equation
        M = np.linalg.solve(A,B)
        return cls(M)

    
    def transform_point(self, source_point):
        """
        This function transforms a source point to destination point.
        Inputs: 
            source_points: pixel locations of a point as an array [x,y]
        """
        M = self.M
        x,y = source_point
        x_new = M[0,0]*x**2 + M[1,0]*y**2 + M[2,0]*x*y + M[3,0]*x + M[4,0]*y + M[5,0]
        y_new = M[0,1]*x**2 + M[1,1]*y**2 + M[2,1]*x*y + M[3,1]*x + M[4,1]*y + M[5,1]

        return np.array([x_new, y_new])
    

