import json
import numpy as np

class FitPlane:
    
    """ Begin constractor methods """
    def __init__():
 
    
    @classmethod
    def from_fitting_points_between_fluorescence_image_and_template(cls, 
        source_image_points, dest_image_points):
        """
        This function initialize FitPlane by points on both source and destination images.
        
        INPUTS:
            source_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
            dest_image_points: pixel location of points of interest as an array or numpy array [[x0,y0],[x1,y1],...]
        """
