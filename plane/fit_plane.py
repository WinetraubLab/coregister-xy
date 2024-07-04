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
    
    def transform_image(self, source_image):
        """
        Transform an image. 
        Inputs:
            source_image: An OpenCV image to be transformed.
        """
        y_range, x_range,_ = source_image.shape
        new_image = np.zeros_like(source_image)

        # Mask to track pixels that need interpolation 
        mask = np.ones((y_range, x_range), dtype=bool)

        # Direct mapping
        for x_i in range(x_range):
            for y_i in range(y_range):
                x_new, y_new = self.transform_point([x_i,y_i])
                x_new = round(x_new)
                y_new = round(y_new)
                if 0 <= y_new < y_range and 0 <= x_new < x_range:
                    new_image[y_new,x_new] = source_image[y_i, x_i]
                    mask[y_new, x_new] = False

        print(np.any(mask))

        # Interpolate unassigned pixels
        while np.any(mask):
            for y_i in range(y_range):
                for x_i in range(x_range):
                    if mask[y_i, x_i]:
                        new_image[y_i, x_i] = self.bilinear_interpolate_pixel(new_image, y_i, x_i)
                        mask[y_i, x_i] = False

        return new_image

    def bilinear_interpolate_pixel(self, img, x, y):
        # Bilinear interpolation to remove missing black "gaps" in transformed image
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, img.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, img.shape[0] - 1)

        Ia = img[y0, x0]
        Ib = img[y1, x0]
        Ic = img[y0, x1]
        Id = img[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id
