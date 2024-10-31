import numpy as np
# from plane.fit_plane import FitPlane
from plane.parse_xml import ParseXML
import pandas as pd

class FitMultiPlane:
    def __init__(self, fitplanes, real_centers, template_size, um_per_pixel):
        """
        - list of barcodes contained in object
            - contains self.scale, self.theta_deg, self.shear_magnitude, self.shear_vector, self.tx, self.ty, self.z
            - Real positions of each barcode center as defined by photobleach script
        - Adjacency matrix to store distances between pairs of barcodes.
        - ??? to store angles between 3 barcodes for xy, xz, yz planes 
        """
        self.fitplanes = fitplanes
        self.real_centers = real_centers # in um
        self.template_size = template_size
        self.um_per_pixel = um_per_pixel
        self.fitplane_centers = self.calc_fitplane_centers() # in um
        self.distances = self.calc_distances() # in um
        self.u = None
        self.v = None
        self.h = None

    @classmethod
    def from_aligned_fitplanes(cls, fitplanes_list, real_centers_list, template_size=401, um_per_pixel=2):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 
            - Individual barcodes: xy center position, rotation angle on xy plane, shear, scaling
            - Distances and angles between barcodes

        Inputs:
            - fitplanes: array-like of fitted planes (ParseXML objects)
            - real_centers: array-like with the locations of the landmarks as defined by photobleach script
            - template_size: square edge dimension of the template in pixels
            - um_per_pixel: number of um represented in each pixel by the template image
        """
        return cls(fitplanes_list, real_centers_list, template_size, um_per_pixel)

    def __len__(self):
        return len(self.fitplanes)
    
    def _check_u_v_consistency_assumptions(self, skip_value_cheks=False):
        """ Check u,v assumptions """
        
        # Skip
        if skip_value_cheks:
            return
    
        # Check u and v are orthogonal and have the same norm
        if not (np.abs(self.u_norm_mm() - self.v_norm_mm())/self.v_norm_mm() < 0.05):
            raise ValueError('u and v should have the same norm')
        if not (np.dot(self.u,self.v)/(self.u_norm_mm()*self.v_norm_mm()) < 0.05):
            raise ValueError('u must be orthogonal to v')
        
        # Check that u vec is more or less in the x-y plane
        min_ratio = 0.15
        slope = abs(self.u[2]) / np.linalg.norm(self.u[:2])
        if not ( slope < min_ratio):
            raise ValueError(
                'Make sure that tissue surface is parallel to x axis. Slope is %.2f (%.0f deg) which is higher than target <%.2f slope'
                % (slope, np.degrees(np.arcsin(slope)),min_ratio))
    
    def calc_distances(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        Units are in um.
        """
        pass

    def calc_fitplane_centers(self):
        """
        Convert Fitplane centers in pixels to distance in um, and add z coordinate for each.
        """
        centers  = [(project.tx + self.template_size/2, project.ty + self.template_size/2) for project in self.fitplanes]
        centers = np.array(centers)
        avgscale = np.mean([project.scale for project in self.fitplanes])
        centers = centers * (self.um_per_pixel / avgscale) # convert fluorescent units from pixels to um
        zs = np.array([project.z for project in self.fitplanes])
        centers_z = np.column_stack((centers, zs))
        return centers_z

    def fit_mapping_to_xy(self):
        """
        Calculate a mapping to project pixels from the angled tissue slice onto a flat plane (match with the photobleach template).
        UVH mapping: for a point (u,v,z') on the sliced tissue, [x,y,z] = vec_u * u + vec_v * v + vec_h
        Prints and returns vectors U, V, and H.
        """
        u = np.array([x[0] for x in self.fitplane_centers])
        v = np.array([x[1] for x in self.fitplane_centers])

        x = np.array([x[0] for x in self.real_centers])
        y = np.array([x[1] for x in self.real_centers])
        z = np.ones_like(y)

        # Number of points
        n = u.shape[0]

        A = np.zeros((3 * n, 9))
        for i in range(n):
            A[3 * i] = [u[i], v[i], 1, 0, 0, 0, 0, 0, 0]      # x equation
            A[3 * i + 1] = [0, 0, 0, u[i], v[i], 1, 0, 0, 0]  # y equation
            A[3 * i + 2] = [0, 0, 0, 0, 0, 0, u[i], v[i], 1]  # z equation

        # Output vector b
        b = np.zeros(3 * n)
        for i in range(n):
            b[3 * i] = x[i]
            b[3 * i + 1] = y[i]
            b[3 * i + 2] = z[i]

        # Solve using least squares
        M, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None) # TODO check that lol
        ux, vx, hx, uy, vy, hy, uz, vz, hz = M

        self.u = np.array([ux, uy, uz])
        self.v = np.array([vx, vy, vz])
        self.h = np.array([hx, hy, hz])

        return self.u, self.v, self.h

    def get_xyz_from_uv(self, point_pix):
        """ Get the 3D physical coordinates of a specific pixel in the image [u_pix, v_pix] """
        u_pix = point_pix[0]
        v_pix = point_pix[1]
        return (self.u*u_pix + self.v*v_pix + self.h)
    
    def get_uv_from_xyz(self, point_mm):
        """ Get the u,v coordinates on an image from a point in space, if point is outside the plane, return the u,v of the closest point. point_mm is a 3D numpy array or array """
	
        point_mm = np.array(point_mm)        

        # Assuming u is orthogonal to v (as it should) for this function to work
        self._check_u_v_consistency_assumptions()
        
        u_hat = self.u_direction()
        u_norm = self.u_norm_mm()
        u_pix = np.dot(point_mm-self.h,u_hat)/u_norm
        
        v_hat = self.v_direction()
        v_norm = self.v_norm_mm()
        v_pix = np.dot(point_mm-self.h,v_hat)/v_norm
        
        return np.array([u_pix, v_pix])
    
    def u_norm_mm(self):
        """ Return the size of pixel u in mm """
        return np.linalg.norm(self.u)
    
    def v_norm_mm(self):
        """ Return the size of pixel v in mm """
        return np.linalg.norm(self.v)
    
    def u_direction(self):
        """ Return a unit vector in the direction of u """
        return self.u / self.u_norm_mm()
        
    def v_direction(self):
        """ Return a unit vector in the direction of v """
        return self.v / self.v_norm_mm()
        
    def normal_direction(self):
        """ Return a unit vector in the direction of the normal """
        return np.cross(self.u_direction(), self.v_direction())
        
    def get_plane_equation(self):
        """ Convert u,v,h to a plane equation ax+by+cz-d=0.
        a,b,c are unitless and normalized a^2+b^2+c^2=1 and d has units of mm """
        normal_vec = self.normal_direction()
        a, b, c = normal_vec
        d = -np.dot(normal_vec, self.h)
        return a,b,c,d

    def print_single_plane_stats(self):
        """
        Prints stats for each FitPlane as a table: shrinkage, rotation, shear, and mean/stdev for each
        Units: um
        """
        num_templates = len(self)
        projects_data = {
        "Template ID": [i for i in range(1, num_templates+1)],
        "Patch Number": [project.template_id for project in self.fitplanes],
        "Z (um)": [project.z for project in self.fitplanes],
        "Center (x)": [project.tx + self.template_size/2 for project in self.fitplanes],
        "Center (y)": [project.ty + self.template_size/2 for project in self.fitplanes],
        "Rotation (deg)": [project.theta_deg for project in self.fitplanes],
        "Scaling": [project.scale for project in self.fitplanes],
        "Shear magnitude": [project.shear_magnitude for project in self.fitplanes],
        "Shear vector (x)": [project.shear_vector[0] for project in self.fitplanes],
        "Shear vector (y)": [project.shear_vector[1] for project in self.fitplanes]
        }

        columns_to_summarize = ["Z (um)", "Rotation (deg)", "Scaling", "Shear magnitude", "Shear vector (x)", "Shear vector (y)"]

        # Create DataFrame
        df = pd.DataFrame(projects_data)

        # Compute mean and standard deviation for selected columns only
        mean_row = df[columns_to_summarize].mean()
        std_row = df[columns_to_summarize].std()

        # Append mean and std as new rows for selected columns only
        summary_df = df.copy()
        summary_df.loc['Mean', columns_to_summarize] = mean_row
        summary_df.loc['StDev', columns_to_summarize] = std_row
        summary_df = summary_df.round(2)
        summary_df = summary_df.replace(np.nan, '', regex=True)

        print("Stats for individual barcodes:\n")
        print(summary_df)

    def project_centers_onto_flat_plane(self):
        """
        Project each (u, v, z') pair from the barcode centers on an angled tissue slice onto a flat xy plane using the 
        vectors UVH from fit_mapping_to_xy. Z for the flat plane is 0.
        Returns: array-like of transformed points.
        """
        pass
    
    def compute_avg_projection_error(a, b):
        """
        Returns the mean distance between points in arrays a and b, for evaluating best-fit calculated projection.
        Ignores z coordinate.
        """
        pass
