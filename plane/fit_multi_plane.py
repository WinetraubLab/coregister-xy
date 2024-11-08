import numpy as np
# from plane.fit_plane import FitPlane
from plane.parse_xml import ParseXML
import pandas as pd

class FitMultiPlane:
    def __init__(self, fitplanes, target_centers, template_size, um_per_pixel):
        self.fitplanes = fitplanes
        self.target_centers = target_centers # in um
        self.template_size = template_size
        self.um_per_pixel = um_per_pixel
        self.fitplane_centers = self.calc_fitplane_centers() # in um
        self.distances = self.set_adjacency_matrix() # in um
        self.u = None
        self.v = None
        self.h = None

    @classmethod
    def from_aligned_fitplanes(cls, fitplanes_list, target_centers_list, template_size=401, um_per_pixel=2):
        """
        Function to calculate/store the params for individual barcodes and combinations of barcodes. 

        :param fit_templates_list: list of barcodes contained in this FitPlane. Each is a FitTemplate object.
        :param target_centers_list: theoretical positions of each barcode center as defined by photobleach script.
        :param template_size: square edge length of the template image used for alignment in each FitTemplate, in pixels.
        :param um_per_pixel: um per pixel in the template image.
        :returns: Initializes an instance of a FitPlane.
        """
        return cls(fitplanes_list, target_centers_list, template_size, um_per_pixel)

    def __len__(self):
        return len(self.fitplanes)
    
    def set_adjacency_matrix(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        Units are in um.
        :returns: adjacency matrix representing distances in um between each pair of barcodes.
        """
        n = len(self.fitplanes)
        adj = np.zeros((n,n))
        for i in range(0, n):
            for j in range(0,n):
                dist = np.sqrt((self.fitplanes[i].tx - self.fitplanes[j].tx)**2 + (self.fitplanes[i].ty - self.fitplanes[j].ty)**2)
                adj[i,j] = dist
        return adj

    def calc_fitplane_centers(self):
        """
        :returns: FitTemplate centers in um, with z coordinate.
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

        :returns: vectors U, V, and H.
        """
        u = np.array([x[0] for x in self.target_centers])
        v = np.array([x[1] for x in self.target_centers])

        x = np.array([x[0] for x in self.target_centers])
        y = np.array([x[1] for x in self.target_centers])
        z = np.array([x[2] for x in self.fitplane_centers])

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
        M, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None) 
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
        
    def get_plane_equation(self):
        """ Convert u,v,h to a plane equation ax+by+cz-d=0.
        a,b,c are unitless and normalized a^2+b^2+c^2=1 and d has units of mm """
        cross = np.cross(self.u, self.v)
        normal_vec = cross / np.linalg.norm(cross)
        a, b, c = normal_vec
        d = -np.dot(normal_vec, self.h)
        return a,b,c,d

    def get_single_template_stats(self):
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
        "Shear unit vector (x)": [project.shear_vector[0] for project in self.fitplanes],
        "Shear unit vector (y)": [project.shear_vector[1] for project in self.fitplanes]
        }

        columns_to_summarize = ["Z (um)", "Rotation (deg)", "Scaling", "Shear magnitude", "Shear unit vector (x)", "Shear unit vector (y)"]

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

        return summary_df

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
