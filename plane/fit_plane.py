import numpy as np
from plane.fit_template import FitTemplate

class FitPlane:
    def __init__(self, uv_px, target_xyz_um, avg_fit_template_scale_factor, template_size, um_per_pixel):
        self.uv_px = uv_px
        self.target_xyz_um = target_xyz_um # in um
        self.template_size = template_size
        self.um_per_pixel = um_per_pixel
        self.uv_um = self._template_centers_px_to_um(avg_fit_template_scale_factor) # in um
        self.u = None
        self.v = None
        self.h = None

        # To find best fit plane:
        p2 = np.hstack((np.array(self.target_xyz_um), np.expand_dims(self.uv_um[:,-1], axis=1))) # add z coords to target centers
        self.best_fit_target_xyz_um = self._find_points_projection_on_best_fit_plane(p2) # z values lie on plane of best fit

    @classmethod
    def from_aligned_fit_templates(cls, uv_px, xyz_um, avg_fit_template_scale_factor, template_size=401, um_per_pixel=2):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 

        :param uv_px: (u,v,w) list of coordinate centers for fitted templates. (u,v) are in pixels and w is the user-assigned z depth of each barcode.
        :param xyz_um: coordinates where each barcode center should be, as defined by photobleach script. Units are in um.
        :param avg_fit_template_scale_factor: average scaling factor across all fitted templates used for alignment. Used in the conversion
            of units from pixels to um.
        :param template_size: square edge length of the template image used for alignment in each FitTemplate, in pixels.
        :param um_per_pixel: um per pixel in the template image.
        :returns: Initializes an instance of a FitPlane.
        """
        return cls(uv_px, xyz_um, avg_fit_template_scale_factor, template_size, um_per_pixel)

    def __len__(self):
        return len(self.uv_px)
    
    def _find_points_projection_on_best_fit_plane(self, points):
        """
        Use this function to calculate the (x, y, z') coordinate for each (x, y, z),
        where z' lies on the best fit plane through these points.

        :param points: A numpy array of shape (n, 3) containing points (x, y, z).
        :returns: projected_points: An array of (x,y,z').
        """
        pass

    def _template_centers_px_to_um(self, avg_scale):
        """
        Converts an array-like of points from pixel positions to um.

        :param avg_scale: Average scale factor to convert from pixels to um, calculated from the scale values stored in each fit_template object.
        :returns: FitTemplate centers in um, with z coordinate.
        """
        centers  = np.array(self.uv_px)[:,:-1]
        centers = centers * (self.um_per_pixel / avg_scale) # convert fluorescent units from pixels to um

        zs = np.array(self.uv_px)[:,-1]
        centers_z = np.column_stack((centers, zs))
        return centers_z

    def fit_mapping_uv_to_xyz(self):
        """
        Calculate a mapping to project pixels from the angled tissue slice onto a flat plane (match with the photobleach template).
        UVH mapping: for a point (u,v,w) on the sliced tissue, [x,y,z] = vec_u * u + vec_v * v + vec_h

        :returns: vectors U, V, and H.
        """
        pass

    def get_xyz_from_uv(self, point_pix):
        """ 
        Get the 3D physical coordinates of a specific pixel in the image [u_pix, v_pix] 
        
        :returns: a single (x,y,z) point 
        """
        pass

    def get_plane_equation(self):
        """ 
        Convert u,v,h to a plane equation ax+by+cz-d=0.
        a,b,c are unitless and normalized a^2+b^2+c^2=1 and d has units of mm 

        :returns: equation coefficients a, b, c, d
        """
    
    def avg_in_plane_projection_error(self):
        """
        Compute average error from projecting (u,v) points to (x,y).
        
        :returns: the average in-plane error, ignoring z error.
        """
        pass

    def avg_out_of_plane_projection_error(self):
        """
        Compute average error between user-specified z coordinates and best fit plane z' coordinates for each (x,y).

        :returns: the average out of plane error, ignoring xy error.
        """
        pass
