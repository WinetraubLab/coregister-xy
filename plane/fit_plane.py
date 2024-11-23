import json
import numpy as np
from scipy.optimize import minimize

class FitPlane:
    
    """ Begin constractor methods """
    def __init__(self,u=None,v=None,h=None):
        self.u = np.array(u)
        self.v = np.array(v)
        self.h = np.array(h)

        # if u is not None and v is not None and h is not None:
        #     self._check_u_v_consistency_assumptions()
    
    @classmethod
    def from_template_centers(cls, template_center_positions_uv_pix, template_center_positions_xyz_um, print_inputs = False):
        """
        This function initializes a FitPlane by a list of points for template centers.

        INPUTS:
            template_center_positions_uv_pix: For each photobleach barcode, find the center position in pixels. This is an
                array of these center points [[x1, y1], [x2, y2],..., [xn, yn]] with shape (n,2)
            template_center_positions_xyz_um: An array [[x1, y1, z1],..., [xn, yn, zn]] of shape (n,3) containing points defining the 
                position (in um) of the locations that each of the points in template_center_positions_uv_pix should map to. 
                These points can be obtained from the photobleaching script.
            print_inputs: prints to screen the inputs of the function for debug purposes.
        """
        fp = cls()

        template_center_positions_uv_pix = np.array(template_center_positions_uv_pix)
        template_center_positions_xyz_um = np.array(template_center_positions_xyz_um)

        # Print inputs
        if print_inputs:
            txt = 'FitPlane.from_template_centers('
            txt = txt + json.dumps(template_center_positions_uv_pix.tolist()) + ','
            txt = txt + json.dumps(template_center_positions_xyz_um.tolist()) + ')'
            print(txt)

        # Input check
        if (template_center_positions_uv_pix.shape[0] != template_center_positions_xyz_um.shape[0]):
            raise ValueError("Number of points should be the same between " + 
                "template_center_positions_uv_pix, template_center_positions_xyz_um")
        
        fp._fit_from_templates(
            template_center_positions_uv_pix, 
            template_center_positions_xyz_um)
                
        # if fp.u is not None:
        #     fp._check_u_v_consistency_assumptions()

        return fp
        
    @classmethod
    def _from_json(self, json_str):
        data = json.loads(json_str)
        self.u=data['u'],
        self.v=data['v'],
        self.h=data['h'],
        
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
    
    """ End constructor methods """    
    def _fit_from_templates(self, template_center_positions_uv_pix, template_center_positions_xyz_um):
        """
        Calculate a mapping with vectors u, v, h to project points from uv coordinates to xyz physical locations.
        """
        u = np.array([x[0] for x in template_center_positions_uv_pix])
        v = np.array([x[1] for x in template_center_positions_uv_pix])

        x = np.array([x[0] for x in template_center_positions_xyz_um])
        y = np.array([x[1] for x in template_center_positions_xyz_um])
        z = np.array([x[2] for x in template_center_positions_xyz_um])

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
              
    def _check_u_v_consistency_assumptions(self, skip_value_cheks=False):
        return
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

        # Assuming u is orthogonal to v (as it shuld) for this function to work
        # self._check_u_v_consistency_assumptions()
        
        u_hat = self.u_direction()
        u_norm = self.u_norm_mm()
        u_pix = np.dot(point_mm-self.h,u_hat)/u_norm
        
        v_hat = self.v_direction()
        v_norm = self.v_norm_mm()
        v_pix = np.dot(point_mm-self.h,v_hat)/v_norm
        
        return np.array([u_pix, v_pix])
        
    def get_fit_plane_xy_projection(self, min_x_mm=None, max_x_mm=None, min_y_mm=None, max_y_mm=None):
        """ When lookin at the pattern from above, return two points on the fit plane that form a line. 
        The line would go from point1 --> point2 where u value increases.
        
        USAGE:
            pt1, pt2 = get_fit_plane_xy_projection()
            pt1, pt2 = get_fit_plane_xy_projection(min_x_mm = 0, max_x_mm = 10, min_y_mm = 0, max_y_mm = 10)
            
        INPUTS:
            If none of the optional inputs are defined then line will be (u,v): (0,0) --> (c_u,c_v)
            If min_x_mm, max_x_mm are defined pt1[0] = min_x_mm, pt2[0] = max_x_mm. 
            If min_y_mm, max_y_mm are defined pt1[1] = min_y_mm, pt2[1] = max_y_mm. 
            If both sets of x and y are defined, we will use the outmost inclusive set
        OUTPUTS:
            (pt1, pt2) where each pt is [x,y,z]
        """
        # Get the points on the plane that satisfy the x condition
        no_x_limit = min_x_mm is None or max_x_mm is None
        if no_x_limit:
            # No clear user limits
            pt1_u_x = np.inf
            pt2_u_x = -np.inf
        else:
            # We need to find where min_x_mm, max_x_mm are on the plane.
            # To do so, we get the equation ax+by+cz+d=0, and set x to the limits, and z to 0 to find y.
            a,b,c,d = self.get_plane_equation()
            min_x_y_mm = -(d+a*min_x_mm)/b
            max_x_y_mm = -(d+a*max_x_mm)/b
            
            # Find u,v on that plane
            tmp1 = self.get_uv_from_xyz([min_x_mm, min_x_y_mm, 0])
            tmp2 = self.get_uv_from_xyz([max_x_mm, max_x_y_mm, 0])
            pt1_u_x, pt2_u_x = min(tmp1[0],tmp2[0]), max(tmp1[0],tmp2[0])
        
        # Get the points on the plane that satisfy the y condition
        no_y_limit = min_y_mm is None or max_y_mm is None
        if no_y_limit:
            # No clear user limits
            pt1_u_y = np.inf
            pt2_u_y = -np.inf
        else:
            # We need to find where min_y_mm, max_y_mm are on the plane.
            # To do so, we get the equation ax+by+cz+d=0, and set y to the limits, and z to 0 to find x.
            a,b,c,d = self.get_plane_equation()
            min_y_x_mm = -(d+b*min_y_mm)/a
            max_y_x_mm = -(d+b*max_y_mm)/a
            
            # Find u,v on that plane
            tmp1 = self.get_uv_from_xyz([min_y_x_mm, min_y_mm, 0])
            tmp2 = self.get_uv_from_xyz([max_y_x_mm, max_y_mm, 0])
            pt1_u_y, pt2_u_y = min(tmp1[0],tmp2[0]), max(tmp1[0],tmp2[0]) 

        if no_x_limit and no_y_limit:
            # No limits found, use default values 
            pt1_u, pt2_u = min(0, self.recommended_center_pix[0]), max(0, self.recommended_center_pix[0])
        else:        
            # Aggregate all points to find the maximum bounds
            pt1_u = min(pt1_u_x,pt1_u_y)
            pt2_u = max(pt2_u_x,pt2_u_y)
        pt12_v = self.recommended_center_pix[1]
        
        # Figure out u,v on the plane that the points correspond to
        pt1 = self.get_xyz_from_uv([pt1_u, pt12_v])
        pt2 = self.get_xyz_from_uv([pt2_u, pt12_v])
        
        return (pt1[:2],pt2[:2])
        
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
        
    def _from_json(self, json_str):
        data = json.loads(json_str)
        self.u=data['u'],
        self.v=data['v'],
        self.h=data['h'],
        self.recommended_center_pix=data['recommended_center_pix']
        
    def to_json(self):
        """
        Serialize the object to a JSON string.
        """
        # Convert the object's dictionary to JSON
        return json.dumps({
            'u': self.u.tolist(),
            'v': self.v.tolist(),
            'h': self.h.tolist(),
            'recommended_center_pix': self.recommended_center_pix.tolist()
            })

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