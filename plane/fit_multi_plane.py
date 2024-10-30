import numpy as np
from plane.fit_plane import FitPlane
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
    
    def calc_distances(self):
        """
        Set adjacency matrix for the list of barcodes given, using their tx ty params.
        Units are in um.
        """
        ## in pixels:
        # n = len(self.fitplanes)
        # adj = np.zeros((n,n))
        # for i in range(0, n):
        #     for j in range(0,n):
        #         dist = np.sqrt((self.fitplanes[i].tx - self.fitplanes[j].tx)**2 + (self.fitplanes[i].ty - self.fitplanes[j].ty)**2)
        #         adj[i,j] = dist
        # return adj
    
        # in um
        n = len(self.fitplanes)
        adj = np.zeros((n,n))
        for i in range(0, n):
            for j in range(0,n):
                dist = np.sqrt((self.fitplane_centers[i, 0] - self.fitplanes[j,0])**2 + (self.fitplanes[i, 1] - self.fitplanes[j, 1])**2)
                adj[i,j] = dist
        return adj

    def calc_fitplane_centers(self):
        """
        Convert Fitplane centers to distance in um, and add z coordinate for each.
        """
        centers  = [(project.tx + self.template_size/2, project.ty + self.template_size/2) for project in self.fitplanes]
        centers = np.array(centers)
        avgscale = np.mean([project.scale for project in ps])
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

        print("Transformation coefficients:")
        print("Vector u:", [ux, uy, uz])
        print("Vector v:", [vx, vy, vz])
        print("Vector h:", [hx, hy, hz])

        return np.array([ux, uy, uz]), np.array([vx, vy, vz]), np.array([hx, hy, hz])

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
        u_coords = self.fitplane_centers[:, 0]
        v_coords = self.fitplane_centers[:, 1]
        
        # Project each (u, v) pair using the vectors u, v, h
        u,v,h = self.fit_mapping_to_xy()
        projected = np.outer(u_coords, u) + np.outer(v_coords, v) + h
        return projected
    
    def compute_avg_projection_error(a, b):
        """
        Returns the mean distance between points in arrays a and b, for evaluating best-fit calculated projection.
        Ignores z coordinate.
        """
        assert a.shape == b.shape, "Input arrays must have the same shape"
        if a.shape[1] == 3:
            a = a[:, :2]
            b = b[:, :2]
        return np.mean(np.linalg.norm(a - b, axis=1))
