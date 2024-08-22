import json
import numpy as np
import cv2 as cv

class FitPlane:
    
    """ Begin constractor methods """
    def __init__(self, M=None, M_rev=None, order=None):
        self.M = M # Transformation from source image to dest image coordinates
        self.M_rev = M_rev # Reverse transformation
        self.order = order
    
    @classmethod
    def from_fitting_points_between_fluorescence_image_and_template(cls, 
        source_image_points, dest_image_points, order=1):
        """
        This function initialize FitPlane by points on both source and destination images.
        
        INPUTS:
            source_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
            dest_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
            order: 1 for affine, 2 for quadratic.
        """
        
        # Type conversion and shape test
        source_image_points = np.array(source_image_points, dtype=np.float32)
        dest_image_points = np.array(dest_image_points, dtype=np.float32)
        assert source_image_points.shape == dest_image_points.shape, "Number of points must match"
        n_points = source_image_points.shape[0]
        if order == 2:
            assert n_points >= 6, "Quadratic transformation must have at least 6 points"
        elif order == 1:
            assert n_points >= 3, "Linear transformation must have at least 3 points"

        def create_map(source_image_points, dest_image_points):
            # Create least squared matrix equations for quadratic transformation
            A = []
            B = []
            for i in range(n_points):
                x_dest, y_dest = dest_image_points[i]
                x, y = source_image_points[i]
                if order == 1:
                    A.append([x,y,1])
                    B.append([x_dest, y_dest])
                elif order == 2:
                    A.append([x**2,y**2,x*y,x,y,1])
                    B.append([x_dest, y_dest])
            B = np.array(B)
            A = np.array(A, dtype=np.float32)
            assert B.shape == (n_points,2), "Shape of matrix B incorrect"
            assert A.shape == (n_points, order*3), "Shape of matrix A incorrect"

            # Solve least squared equation
            M, residuals, rank, s = np.linalg.lstsq(A,B, rcond=None)
            return M
    
        return cls(
            M = create_map(source_image_points, dest_image_points),
            M_rev = create_map(dest_image_points, source_image_points),
            order=order
        )
    
    def transform_point(self, source_point, reverse=False):
        """
        This function transforms a source point to destination point.
        Inputs: 
            source_points: pixel locations of a point as an array [x,y]
        """
        if reverse:
            M = self.M_rev
        else:
            M = self.M
        x,y = source_point
        if self.order == 2:
            x_new = M[0,0]*x**2 + M[1,0]*y**2 + M[2,0]*x*y + M[3,0]*x + M[4,0]*y + M[5,0]
            y_new = M[0,1]*x**2 + M[1,1]*y**2 + M[2,1]*x*y + M[3,1]*x + M[4,1]*y + M[5,1]
        elif self.order == 1:
            x_new = M[0,0]*x + M[1,0]*y + M[2,0]
            y_new = M[0,1]*x + M[1,1]*y + M[2,1]

        return np.array([x_new, y_new])
    
    def transform_image(self, source_image, dest_image_shape=None, reverse=False):
        """
        Transform an image. 
        Inputs:
            source_image: An OpenCV image to be transformed.
            dest_image_shape: touple of image size (pixels) of the dest image.
                If set to None, will match source_image shape.
        """

        # Figure out what is the dest image size
        if dest_image_shape is None:
            dest_image_shape = source_image.shape
        elif len(dest_image_shape) < len(source_image.shape):
            # User didn't define color channel in the dest image size vector
            dest_image_shape = dest_image_shape + (source_image.shape[2],)
        
        # Find reverse mapping 
        transformed_coords = -np.ones((dest_image_shape[0], dest_image_shape[1],2), dtype=np.float32)
        y_dest_range, x_dest_range, _ = dest_image_shape
        for x_i in range(x_dest_range):
            for y_i in range(y_dest_range):
                if reverse:
                    x_source, y_source = self.transform_point([x_i, y_i], False)
                else:
                    x_source, y_source = self.transform_point([x_i, y_i], True)
                if 0 <= round(y_source) < source_image.shape[0] and 0 <= round(x_source) < source_image.shape[1]:
                    transformed_coords[y_i, x_i] = [x_source, y_source]

        dest_image = cv.remap(source_image, transformed_coords, None, interpolation=cv.INTER_LINEAR)
        return dest_image
    
    def compute_physical_params(self, reverse=False):
        """
        Compute physical representation of transform from matrix M.
        Returns:
            translation (x,y), rotation, scaling x y, and shear.
            Shear is horizontal (x-direction).
        """
        M = np.transpose(self.M)
        if reverse:
            M = np.transpose(self.M_rev)
        
        a, b, tx = M[0]
        c, d, ty = M[1]

        translation = (tx, ty)
        theta_deg = np.degrees(np.arctan2(c,a))
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)



        # Rotation matrix
        R = np.array([
            [np.cos(np.arctan2(c,a)), -np.sin(np.arctan2(c,a))],
            [np.sin(np.arctan2(c,a)), np.cos(np.arctan2(c,a))]
        ])

        # Compute the inverse rotation matrix
        R_inv = np.linalg.inv(R)

        # Shear in x-direction (after removing rotation effect)
        shear_x = R_inv[0, 0] * b + R_inv[0, 1] * c

        # Shear in y-direction (after removing rotation effect)
        shear_y = R_inv[1, 0] * b + R_inv[1, 1] * c


        return translation, theta_deg, scale_x, scale_y, shear_x, shear_y
    