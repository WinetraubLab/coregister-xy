import json
import numpy as np
from scipy.optimize import minimize

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
        cls, template_center_positions_uv_pix, template_center_positions_xyz_um,
        forced_plane_normal = None, 
        print_inputs = False):
        """
        This function initializes a FitPlane by a list of points for template centers.

        INPUTS:
            template_center_positions_uv_pix: For each photobleach barcode, find the center position in pixels. This is an
                array of these center points [[x1, y1], [x2, y2],..., [xn, yn]] with shape (n,2)
            template_center_positions_xyz_um: An array [[x1, y1, z1],..., [xn, yn, zn]] of shape (n,3) containing points defining the 
                position (in um) of the locations that each of the points in template_center_positions_uv_pix should map to. 
                These points can be obtained from the photobleaching script.
            forced_plane_normal: When set to a 3D vector, will find the best plane fit that also satisfies u cross v is plane norm.
            print_inputs: prints to screen the inputs of the function for debug purposes.
        """

        # Input check
        template_center_positions_uv_pix = np.array(template_center_positions_uv_pix)
        template_center_positions_xyz_um = np.array(template_center_positions_xyz_um)
        if (template_center_positions_uv_pix.shape[0] != template_center_positions_xyz_um.shape[0]):
            raise ValueError("Number of points should be the same between " + 
                "template_center_positions_uv_pix, template_center_positions_xyz_um")
        if template_center_positions_uv_pix.shape[1] != 2:
            raise ValueError("Number of elements in template_center_positions_uv_pix should be two")
        if template_center_positions_xyz_um.shape[1] != 3:
            raise ValueError("Number of elements in template_center_positions_xyz_um should be three")
        if forced_plane_normal is not None:
            forced_plane_normal = np.array(forced_plane_normal)
            if forced_plane_normal.shape[0] != 3:
                raise ValueError("forced_plane_normal should be 3D vector")
            forced_plane_normal = forced_plane_normal / np.linalg.norm(forced_plane_normal)

        # Print inputs
        if print_inputs:
            txt = ("FitPlane.from_template_centers(" +
                   json.dumps(template_center_positions_uv_pix.tolist()) + "," +
                   json.dumps(template_center_positions_xyz_um.tolist()))   
            if forced_plane_normal is not None:
                txt += ',' + json.dumps(forced_plane_norm.tolist())
            txt += ')'
            print(txt)

        # Construct measurement matrix
        u_pt = np.array([x[0] for x in template_center_positions_uv_pix])
        v_pt = np.array([x[1] for x in template_center_positions_uv_pix])
        x_pt = np.array([x[0] for x in template_center_positions_xyz_um])
        y_pt = np.array([x[1] for x in template_center_positions_xyz_um])
        z_pt = np.array([x[2] for x in template_center_positions_xyz_um])

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
            u=data['u'],
            v=data['v'],
            h=data['h'],
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
        return np.cross(self.u_direction(), self.v_direction())
        
    def get_plane_equation(self):
        """ Convert u,v,h to a plane equation ax+by+cz-d=0.
        a,b,c are unitless and normalized a^2+b^2+c^2=1 and d has units of mm """
        normal_vec = self.normal_direction()
        a, b, c = normal_vec
        d = -np.dot(normal_vec, self.h)
        
        return a,b,c,d
        
    def get_xyz_from_uv(self, point_pix):
        """ Get the 3D physical coordinates of a specific pixel in the image [u_pix, v_pix] """
        u_pix = point_pix[0]
        v_pix = point_pix[1]
        return (self.u*u_pix + self.v*v_pix + self.h)
    
    def get_uv_from_xyz(self, point_mm):
        """ Get the u,v coordinates on an image from a point in space, if point is outside the plane, return the u,v of the closest point. point_mm is a 3D numpy array or array """
	
        point_mm = np.array(point_mm)        
        
        u_hat = self.u_direction()
        u_norm = self.u_norm_mm()
        u_pix = np.dot(point_mm-self.h,u_hat)/u_norm
        
        v_hat = self.v_direction()
        v_norm = self.v_norm_mm()
        v_pix = np.dot(point_mm-self.h,v_hat)/v_norm
        
        return np.array([u_pix, v_pix])
        
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

    def get_v_line_fit_plane_intercept(self, line_position_mm):
        """ 
        Returns a,b,c that correspond to the equation a*u+b*v+c=0. 
        u,v are in pixels. a^2+b^2=1
        The equation corresponds to where on the image plane intersects a vertical
        line x=line_position_mm.
        """
    
        # Get equation (ax+by+cz+d=0), make a function to generate a point on plane
        [a,b,c,d] = self.get_plane_equation() 
        def gen_point(z):
            # Auxilary function to generate a point on the plane given arbitrary z
            x = line_position_mm
            y = -(a*x+c*z+d)/b 
            z = z 
            return [x,y,z]

        # Using two arbitrary points, form the output equation
        pt1 = self.get_uv_from_xyz(gen_point(0))
        pt2 = self.get_uv_from_xyz(gen_point(1))
        a_out = pt2[1]-pt1[1]
        b_out = pt2[0]-pt1[0]
        c_out = pt2[0]*pt1[1] - pt1[0]*pt2[1]

        # Normalize
        norm = np.sqrt(a_out**2 + b_out**2)
        return (a_out/norm, b_out/norm, c_out/norm)

    def get_h_line_fit_plane_intercept(self, line_position_mm):
        """ 
        Returns a,b,c that correspond to the equation a*u+b*v+c=0. 
        u,v are in pixels. a^2+b^2=1
        The equation corresponds to where on the image plane intersects a horizontal 
        line y=line_position_mm.
        """
    
        # Get equation (ax+by+cz+d=0), make a function to generate a point on plane
        [a,b,c,d] = self.get_plane_equation() 
        def gen_point(z):
            # Auxilary function to generate a point on the plane given arbitrary z
            y = line_position_mm
            x = -(b*y+c*z+d)/a 
            z = z 
            return [x,y,z]

        # Using two arbitrary points, form the output equation
        pt1 = self.get_uv_from_xyz(gen_point(0))
        pt2 = self.get_uv_from_xyz(gen_point(1))
        a_out = pt2[1]-pt1[1]
        b_out = pt2[0]-pt1[0]
        c_out = pt2[0]*pt1[1] - pt1[0]*pt2[1]

        # Normalize
        norm = np.sqrt(a_out**2 + b_out**2)
        return (a_out/norm, b_out/norm, c_out/norm)
