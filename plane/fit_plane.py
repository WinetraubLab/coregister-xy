import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import cv2

class FitPlane:
    
    """ Begin constractor methods """
    def __init__(self,u_mm=None,v_mm=None,h_mm=None):
        if u_mm is not None and v_mm is not None and h_mm is not None:
            self.u = np.array(u_mm) # mm
            self.v = np.array(v_mm) # mm
            self.h = np.array(h_mm) # mm
        else:
            self.u = None
            self.v = None
            self.h = None
    
    @classmethod
    def from_template_centers(
        cls, template_center_positions_uv_pix, template_center_positions_xyz_mm,
        forced_plane_normal = None, 
        print_inputs = False):
        """
        This function initializes a FitPlane by a list of points for template centers.

        INPUTS:
            template_center_positions_uv_pix: For each photobleach barcode, find the center position in pixels. This is an
                array of these center points [[x1, y1], [x2, y2],..., [xn, yn]] with shape (n,2)
            template_center_positions_xyz_mm: An array [[x1, y1, z1],..., [xn, yn, zn]] of shape (n,3) containing points defining the 
                position (in um) of the locations that each of the points in template_center_positions_uv_pix should map to. 
                These points can be obtained from the photobleaching script.
            forced_plane_normal: When set to a 3D vector, will find the best plane fit that also satisfies u cross v is plane norm.
            print_inputs: prints to screen the inputs of the function for debug purposes.
        """

        # Input check
        template_center_positions_uv_pix = np.array(template_center_positions_uv_pix)
        template_center_positions_xyz_mm = np.array(template_center_positions_xyz_mm)
        if (template_center_positions_uv_pix.shape[0] != template_center_positions_xyz_mm.shape[0]):
            raise ValueError("Number of points should be the same between " + 
                "template_center_positions_uv_pix, template_center_positions_xyz_mm")
        if template_center_positions_uv_pix.shape[1] != 2:
            raise ValueError("Number of elements in template_center_positions_uv_pix should be two")
        if template_center_positions_xyz_mm.shape[1] != 3:
            raise ValueError("Number of elements in template_center_positions_xyz_mm should be three")
        if forced_plane_normal is not None:
            forced_plane_normal = np.array(forced_plane_normal)
            if forced_plane_normal.shape[0] != 3:
                raise ValueError("forced_plane_normal should be 3D vector")
            forced_plane_normal = forced_plane_normal / np.linalg.norm(forced_plane_normal)

        # Print inputs
        if print_inputs:
            txt = ("FitPlane.from_template_centers(" +
                   json.dumps(template_center_positions_uv_pix.tolist()) + "," +
                   json.dumps(template_center_positions_xyz_mm.tolist()))   
            if forced_plane_normal is not None:
                txt += ',' + json.dumps(forced_plane_normal.tolist())
            txt += ')'
            print(txt)

        # Construct measurement matrix
        u_pt = np.array([x[0] for x in template_center_positions_uv_pix])
        v_pt = np.array([x[1] for x in template_center_positions_uv_pix])
        x_pt = np.array([x[0] for x in template_center_positions_xyz_mm])
        y_pt = np.array([x[1] for x in template_center_positions_xyz_mm])
        z_pt = np.array([x[2] for x in template_center_positions_xyz_mm])

        # Number of points
        n = u_pt.shape[0]

        A = np.zeros((3 * n, 9))
        for i in range(n):
            A[3 * i + 0] = [u_pt[i], v_pt[i], 1, 0, 0, 0, 0, 0, 0] # x equation
            A[3 * i + 1] = [0, 0, 0, u_pt[i], v_pt[i], 1, 0, 0, 0] # y equation
            A[3 * i + 2] = [0, 0, 0, 0, 0, 0, u_pt[i], v_pt[i], 1] # z equation

        # Output vector b
        b = np.zeros(3 * n)
        for i in range(n):
            b[3 * i + 0] = x_pt[i]
            b[3 * i + 1] = y_pt[i]
            b[3 * i + 2] = z_pt[i]

        # Define a cost function
        def cost_fun(x):
            x = np.array(x)
            lstsq_error = np.mean((A @ x - b)**2) # This is data reliability error

            ux, vx, hx, uy, vy, hy, uz, vz, hz = x

            u = np.array([ux, uy, uz])
            u_norm = np.linalg.norm(u)
            u_hat = u / u_norm
            v = np.array([vx, vy, vz])
            v_norm = np.linalg.norm(v);
            v_hat = v / v_norm

            if forced_plane_normal is not None: # Normal requirement
                n = np.cross(u_hat,v_hat)
                dot_product_with_direction = np.dot(n, forced_plane_normal)
                normal_error = np.abs(dot_product_with_direction-1)
            else:
                normal_error = 0   
            
            return lstsq_error + normal_error*1       

        # Initial guess, just least square
        x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None) 

        # Solve the constrained optimization problem
        result = minimize(cost_fun, x0)

        # Get the solution vector
        ux, vx, hx, uy, vy, hy, uz, vz, hz = result.x

        fp = cls(
            u_mm = np.array([ux, uy, uz]),
            v_mm = np.array([vx, vy, vz]),
            h_mm = np.array([hx, hy, hz]))

        return fp
        
    @classmethod   
    def from_json(cls, json_str):
        """
        Deserialize a JSON string to a FitPlane object.
        """
        # Parse the JSON string
        data = json.loads(json_str)
        # Create a new FitPlane object using the parsed data
        return cls(
            data['u'],
            data['v'],
            data['h'],
        )
        
    def to_json(self):
        """
        Serialize the object to a JSON string.
        """
        # Convert the object's dictionary to JSON
        return json.dumps({
            'u': self.u.tolist(),
            'v': self.v.tolist(),
            'h': self.h.tolist(),
            })
    
              
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
        n = np.cross(self.u_direction(), self.v_direction())
        n = n / np.linalg.norm(n)
        return n
        
    def get_plane_equation(self):
        """ Convert u,v,h to a plane equation ax+by+cz-d=0.
        a,b,c are unitless and normalized a^2+b^2+c^2=1 and d has units of mm """
        normal_vec = self.normal_direction()
        a, b, c = normal_vec
        d = -np.dot(normal_vec, self.h)
        
        return a,b,c,d

    def get_xyz_from_uv(self, point_pix):
        """ Get the 3D physical coordinates of pixels in the image [u_pix, v_pix] """
        A = np.vstack([self.u, self.v, self.h]).T
        uv_pix = np.array(point_pix)

        if uv_pix.ndim == 1: # Single point. shape (2,) -> (1, 2)
            uv_pix = uv_pix[np.newaxis, :]
            single_point = True
        else:
            single_point = False

        # Add homogeneous coordinate (N, 2) -> (N, 3)
        ones = np.ones((uv_pix.shape[0], 1))
        points_mat = np.hstack([uv_pix, ones]) 

        xyz_coords = points_mat @ A.T 

        if single_point:
            return xyz_coords[0]  # Return (3,) for single input
        return xyz_coords  # Return (N, 3)
    
    def get_uv_from_xyz(self, point_mm):
        """ Get the UV pixel coordinates of points in 3D xyz space"""
        A = np.vstack([self.u, self.v, self.h]).T  
        A_inv = np.linalg.inv(A)

        point_mm = np.asarray(point_mm)

        if point_mm.ndim == 1: # Single point: (3,) → (1, 3)
            point_mm = point_mm[np.newaxis, :]
            single_point = True
        else:
            single_point = False

        uv_all = point_mm @ A_inv.T  # Shape: (N, 3)
        uv = uv_all[:, :2]  # Shape: (N, 2)

        if single_point:
            return uv[0]  # Shape: (2,)
        return uv  # Shape: (N, 2)
    
    def distance_from_origin_mm(self):
        """ Compute a signed distance from origin """
        return np.dot(self.h, self.normal_direction())
    
    def xy_rotation_deg(self):
        """ When looking at the top, what is the angle of the plane on the xy plane """
        dot_product = np.dot(self.u_direction(),np.array([1, 0, 0]))
        xy_angle = np.degrees(np.arccos(dot_product))
        return xy_angle
    
    def tilt_deg(self):
        """ What is the tilt of the plane compared to z axis """
        dot_product = np.dot(self.v_direction(),np.array([0, 0, 1]))
        z_angle = np.degrees(np.arccos(dot_product))
        return z_angle  

    def image_to_physical(self, cv2_image,
                          x_range_mm=[-1,1], y_range_mm=[-1,1], pixel_size_mm = 1e-3):
        """
        This function takes a picture (cv2_image) with coordinates u,v and project it to real space within range x_range_mm, y_range_mm.
        Projected image is returned.
        This function assumes that plane normal is more or less perpadicular to xy plane.
        """

        # Input checks
        x_range_mm = np.array(x_range_mm)
        y_range_mm = np.array(y_range_mm)

        # Check that the normal direction is very close to z.
        n = self.normal_direction()
        if np.sqrt(n[0]**2+n[1]**2) > 0.05:
            raise NotImplementedError(f'Normal direction must be close to z. n is [{n[0]},{n[1]},{n[2]}]')

        # Define the edges of the transofmation
        def dest(x,y):
            return( (x-x_range_mm[0])/pixel_size_mm, (y-y_range_mm[0])/pixel_size_mm )

        # Get affine transformation matrix
        uv = np.array([[0,0],[0,1000],[1000,0]])
        xy = np.array([dest(self.get_xyz_from_uv(t)[0], self.get_xyz_from_uv(t)[1]) for t in uv])
        M = cv2.getAffineTransform(np.array(uv).astype(np.float32), (xy).astype(np.float32))

        # Apply the affine transformation using warpAffine
        width, height = dest(x_range_mm[1], y_range_mm[1])
        width = int(width)
        height = int(height)
        transformed_image = cv2.warpAffine(
            cv2_image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return transformed_image

    def get_template_center_positions_distance_metrics(self, uv_pix, xyz_mm, mean=True):
        """ 
        uv_pix: coordinates in pixels, array shape (2,n)
        xyz_mm: coordinates in mm, array shape (3,n)
        mean: whether to average error across all input uv and xyz points.
        Returns in plane and out of plane distances between mapped uv points and corresponding xyz points. 
        Returns scalar average if mean=True, array of distances per point otherwise.
        """
        uv_to_xyz = np.array([self.get_xyz_from_uv(p) for p in uv_pix])
        in_plane = np.sqrt(np.sum((uv_to_xyz[:, :2] - xyz_mm[:, :2])**2, axis=1))
        out_plane = np.abs(uv_to_xyz[:, 2] - xyz_mm[:, 2]) # Avg differences on z
        if mean:
            return np.mean(in_plane), np.mean(out_plane)
        else:
            return in_plane, out_plane
