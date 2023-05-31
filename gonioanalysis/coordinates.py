'''
Working in 3D space
'''
import math
from math import sin, cos, tan, radians, pi, acos, atan, sqrt, degrees, atan2

import numpy as np


def where_vertical_between(points_3d, lower=None, upper=None, reverse=False):
    ''''Returns a boolean array based on points' vertical angle.

    Takes in 3D points and returns an 1D True/False array of length points_3d
    
    Arguments
    ---------
    points_3d : sequence
        A sequence of (x,y,z) points
    lower : float
        Lower vertical angle degree in degrees
    upper : float
    reverse : bool
        If True, inverses the returned array (True -> False and vice versa).

    Returns
    -------
    booleans : ndarray
        1D True/False array
    '''
    
    # Calculate each point's vertical angle in degrees
    verticals = np.degrees(np.arcsin(points_3d[:,2]/ np.cos(points_3d[:,0]) ))

    # Check each point's y coordinate; If it is negative, we
    # have to fix its angle
    # FIXME: Do this in numpy for better performance
    for i_point in range(len(points_3d)):
        if points_3d[i_point][1] < 0:
            if verticals[i_point] > 0:
                verticals[i_point] = 180-verticals[i_point]
            else:
                verticals[i_point] = -180-verticals[i_point]

    booleans = np.ones(len(points_3d), dtype=bool)
    if lower is not None:
        booleans = booleans * (verticals > lower)
    if upper is not None:
        booleans = booleans * (verticals < upper)
    
    if reverse:
        booleans = np.invert(booleans)

    return booleans
 




def to_spherical(x,y,z, return_degrees=False):
    '''
    Transform to spherical coordinates (ISO)
    
    return_degrees     If true, return angles in degrees instead of radians

    Returns: r, phi, theta
    '''
    r = sqrt(x**2+y**2+z**2)
    phi = atan2(y, x)
    theta = acos(z/(r))
    
    if return_degrees:
        phi = degrees(phi)
        theta = degrees(theta)

    return r, phi, theta

def normalize(P0, P1, scale=1):
    '''
    Normalize a vector
    
    P0      Vector start point
    P1      Vector end point
    '''
    P0 = np.array(P0)
    P1 = np.array(P1)


    vec = P1 - P0
    
    if len(np.nonzero(vec)[0]) == 0:
        return P0

    vec = vec / np.linalg.norm(vec)
    vec *= scale
    vec += P0

    return vec


#def mean_distance(points):
#    return np.mean(np.linalg.norm(points, axis=1))




def nearest_neighbour(point_A, points_B, max_distance=None):
    '''
    Return the nearest point to the point_A from points_B.

    point_A         1D np.array [x0, y0, z0]
    points_B        2D np.array [ [x1,y1,z1], [z2,y2,z2], ... ]
    '''

    distances = np.linalg.norm(points_B - point_A, axis=1)

    i_shortest = np.argmin(distances)

    if max_distance:
        if distances[i_shortest] > max_distance:
            return False

    return i_shortest


def mean_vector(point, vectors):
    '''
    Average vectors and return a vector at point point.

    '''

    av = np.mean(vectors, axis=0)
    if np.linalg.norm(av) != 0:

        av += np.array(point)
        av = force_to_tplane(point, av)



        for i in range(0,len(vectors)):
            wanted_len = np.linalg.norm(vectors[i])

            if wanted_len != 0:
                break
        av -= np.array(point)
        av = (av / np.linalg.norm(av) * wanted_len)
    else:
        av = np.array([0,0,0])
        pass
    #x,y,z = point

    #return (angle_tag, (x, av[0]), (y, av[1]), (z, av[2]) )
    return av




def rotate_about_x(point, degs):
    '''
    Rotate a point in 3D space along the first axis (x-axis).
    '''
    
    c = cos(radians(degs))
    s = sin(radians(degs))

    Rx = np.array([[1,0,0], [0, c, -s], [0, s, c]])

    return np.dot(Rx, np.array(point))
    


def force_to_tplane(P0, P1, radius=1):
    '''
    Forces a vector (P0-P1) on a tangent plane of a sphere but
    retaining the vector's length

    P0 is the point on the sphere (and the tangent plane)
    P1 is the point off the tangent plane

    Returns P2, point on the tangent plane and the line connecting
    the sphere centre point to the P1.

    Notice DOES NOT RETURN VEC BUT P2 (vec=P2-P0)
    '''
    

    a = radius / (P0[0]*P1[0]+P0[1]*P1[1]+P0[2]*P1[2])
    
    return P1 * a


#def projection_to_tplane(P)


def camera2Fly(horizontal, vertical, radius=1):
    '''
    With the given goniometer positions, calculates camera's position
    in fly's cartesian coordinate system.
    
    Input in degrees
    '''
    #print('Horizontal {}'.format(horizontal))
    #print('Vertical {}'.format(vertical))

    h = radians(horizontal)
    v = radians(vertical)
    
    #x = sqrt( radius**2 * (1 - (cos(h) * sin(v))**2 * (tan(v)**2+1)) )
    
    #y = cos(h) * cos(v) * radius
    
    #z = cos(h) * sin(v) * radius

    y = cos(h)*cos(v)*radius
    z = y * tan(v)
    
    # abs added because finite floating point precision
    x = sqrt(abs(radius**2 - y**2 - z**2))
    
    # Make sure zero becomes zero
    if x < 10**-5:
        x = 0

    # Obtain right sign for x
    looped = int(h / (2*pi))
    if not 0 < (h - 2*pi*looped ) < pi:
        x = -x


    return x, y, z


def camera_rotation(horizontal, vertical, return_degrees=False):
    '''
    Camera's rotation
    '''
    
    if vertical > 90:
        vvertical = 180-vertical
    else:
        vvertical = vertical

    rot = -(sin(radians(horizontal))) * radians(vvertical)
    #rot= -((radians(horizontal))/(pi/2)) * radians(vvertical)
    
    if vertical > 90:
        rot += radians(180)
        rot = -rot
  
    if return_degrees:
        rot = degrees(rot)

    return -rot



def camvec2Fly(imx, imy, horizontal, vertical, radius=1, normalize=False):
    '''
    Returns 3D vector endpoints.

    normalize       If true, return unit length vectors
    '''
    
    #imx = 0
    #imy = 1

    #rot = (1- cos(radians(horizontal))) * radians(vertical)
    
    #rot = -(radians(horizontal)/(np.pi/2)) * radians(vertical)
    '''  
    rot = camera_rotation(horizontal, vertical)

    cimx = imx * cos(rot) - imy * sin(rot)
    cimy = imx * sin(rot) + imy * cos(rot)
   

    # find the plane

    # coordinates from the plane to 3d

    #dz = cos()

    x,y,z = camera2Fly(horizontal, vertical, radius=radius)
    
    #print('x={} y={} z={}'.format(x,y,z))
    '''
    # Create unit vectors in camera coordinates
    '''
    if x == 0 and y > 0:
        b = pi/2
    elif x== 0 and y < 0:
        b = pi + pi/2
    else:
        b = atan(y/x) # angle in xy-plane, between radial line and x-axis
        #b = atan2(y,x)

    if x < 0:
       b += pi
    e = acos(z/sqrt(x**2+y**2+z**2))# anti-elevation

    if y < 0:
        e += pi

    uimx = np.array([-sin(b) , cos(b), 0])
    #uimy = np.asarray([-cos(e) * sin(b) , - cos(e) * cos(b),  sin(e) ])
    # Fixed this on 6.9.2019
    uimy = np.array([-cos(b) * cos(e) , -sin(b) * cos(e),  sin(e) ])
    '''

    x,y,z = camera2Fly(horizontal, vertical, radius=radius)

    uimx = np.array(camera2Fly(horizontal, vertical, radius=radius)) - np.array(camera2Fly(horizontal+1, vertical, radius=radius))
    uimx = uimx / np.linalg.norm(uimx)

    uimy = np.array([0, -sin(radians(vertical)), cos(radians(vertical))])

    #print('vertical {}'.format(vertical))
    #print('imx is {}'.format(imx))
    #fx, fy, fz = np.array([x,y,z]) + uimx*cimx + uimy*cimy
    vector = uimx*imx + uimy*imy

    if normalize:
        length = np.linalg.norm(vector)
        if length != 0:
            
            if type(normalize) == type(42) or type(normalize) == type(4.2):
                length /= normalize
        
            vector = vector / length


    fx, fy, fz = np.array([x,y,z]) + vector 
    '''
    if normalize:
        uim = uimx*cimx + uimy*cimy
        length = np.sqrt(uim[0]**2 + uim[1]**2 + uim[2]**2)
        
        if length != 0:

            if type(normalize) == type(42) or type(normalize) == type(4.2):
                length /= normalize
        
            fx, fy, fz = np.array([x,y,z]) + (uimx*cimx + uimy*cimy)/length
    '''

    #print("Elevation {}".format(degrees(e)))
    #print('B {}'.format(b))
    #print('uimx {}'.format(uimx))
    #print('uimy {}'.format(uimy))
    #print()

    return fx, fy, fz
#
#def findDistance(point1, point2):
#    '''
#    Returns PSEUDO-distance between two points in the rotation stages angles coordinates (horizontal_angle, vertical_angle).
#
#    It's called pseudo-distance because its not likely the real 3D cartesian distance + the real distance would depend
#    on the radius that is in our angles coordinate system omitted.
#
#    In the horizontal/vertical angle system, two points may seem to be far away but reality (3D cartesian coordinates)
#    the points are closed to each other. For example, consider points
#        (90, 10) and (90, 70)
#    These points are separated by 60 degrees in the vertical (pitch) angle, but because the horizontal angle is 90 degrees
#    in both cases, they are actually the same point in reality (with different camera rotation)
#
#    INPUT ARGUMENTS     DESCRIPTION
#    point1              (horizontal, vertical)
#
#
#    TODO:   - Implement precise distance calculation in 3D coordinates
#            - ASSURE THAT THIS ACTUALLY WORKS???
#    '''
#    # Scaler: When the horizontal angle of both points is close to 90 or -90 degrees, distance
#    # should be very small
#    scaler = abs(math.sin((point1[0] + point2[0])/ 2))
#    # All this is probably wrong, right way to do this is calculate distances on a sphere
#    return scaler * math.sqrt( (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )
#
#
#def findClosest(point1, points, distance_function=None):
#    '''
#    Using findDistance, find closest point to point1.
#    '''
#    
#    distances = []
#
#    if not callable(distance_function):
#        distance_function = findDistance
#
#    for point2 in points:
#        distances.append( distance_function(point1, point2) )
#
#    argmax_i = distances.index(min(distances))
#    
#    return points[argmax_i]
#


def get_rotation_matrix(axis, rot):
    '''
    Returns an elementar rotation matrix.

    axis        'x', 'y', or 'z'
    rot         rotation in radians
    '''
    # Calculate sin and cos terms beforehand
    c = cos(rot)
    s = sin(rot)

    if axis == 'x':
        return np.array([[1,0,0], [0,c,-s], [0,s,c]])
    elif axis == 'y':
        return np.array([[c,0,s], [0,1,0], [-s,0,c]]) 
    elif axis == 'z':
        return np.array([[c,-s,0], [s,c,0], [0,0,1]])
    else:
        raise ValueError('Axis has to be x, y, or z, not {}'.format(axis))


def rotate_along_arbitrary(P1, points, rot):
    '''
    Rotate along arbitrary axis.

    P0 is at origin.
    
    P0 and P1 specify the rotation axis

    Implemented from here:
    http://paulbourke.net/geometry/rotate/
    
    Arguments
    ---------
    P1 : np.ndarray
    points : np.ndarray
    rot : float or int
        Rotation in radians (not degres!)
    '''

    a,b,c = P1 / np.linalg.norm(P1)
    d = math.sqrt(b**2 + c**2)
    
    if d == 0:
        Rx = np.eye(3)
        Rxr = np.eye(3)
    else:
        Rx = np.array([[1,0,0],[0,c/d, -b/d],[0,b/d, c/d]])
        Rxr = np.array([[1,0,0],[0,c/d, b/d],[0,-b/d, c/d]])

    Ry = np.array([[d,0,-a],[0,1,0], [a,0,d]])
    Ryr = np.array([[d,0,a],[0,1,0], [-a,0,d]])

    Rz = get_rotation_matrix('z', rot)

    return (Rxr @ Ryr @ Rz @ Ry @ Rx @ points.T).T


def rotate_points(points, yaw, pitch, roll):
    '''
    Just as rotate_vectors but only for points.
    
    Arguments
    ---------
    yaw, pitch, roll : float or int
        Rotations in radians (not degrees!)
    '''
    yaw_ax = (0,0,1)
    pitch_ax = (1,0,0)
    roll_ax = (0,1,0)

    axes = np.array([yaw_ax, pitch_ax, roll_ax])

    rotations = [yaw, pitch, roll]
    
    for i in range(3):
        points = rotate_along_arbitrary(axes[i], points, rotations[i])
    
    return points



def rotate_vectors(points, vectors, yaw, pitch, roll):
    '''
    In the beginning,  it is assumed that
        yaw     rotation along Z
        pitch   rotation along X
        roll    rotation along Y

    ie that the fly head is at zero rotation, antenna roots pointing towards
    positive y-axis.
    '''

    yaw_ax = (0,0,1)
    pitch_ax = (1,0,0)
    roll_ax = (0,1,0)

    axes = np.array([yaw_ax, pitch_ax, roll_ax])

    rotations = [yaw, pitch, roll]
    
    for i in range(3):
        new_points = rotate_along_arbitrary(axes[i], points, rotations[i])
        new_vectors = rotate_along_arbitrary(axes[i], points+vectors, rotations[i]) - new_points
        
        points = new_points
        vectors = new_vectors


        # Update axes
        #axes = rotate_along_arbitrary(axes[i], axes, rotations[i])
    
    return points, vectors


def distance(a, b):
    '''
    Calculates distance between two points in 3D cartesian space.
    a,b      (x,y,z)
    '''

    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def optimal_sampling(horizontals, verticals):
    '''
    Determine optimal way to sample using two orthogonal goniometers.
    '''
    
    steps = ((horizontals[1]-horizontals[0]), (verticals[1]-verticals[0]))

    min_distance = 0.75 * distance(camera2Fly(steps[0], steps[1]), camera2Fly(0,0))

  
    goniometer_vals = {}
    
    points = []

    for vertical in verticals:
        goniometer_vals[vertical] = []
        for horizontal in horizontals:
            point = camera2Fly(horizontal, vertical)
            
            append = True
            
            for previous_point in points:
                if distance(previous_point, point) < min_distance:
                    append = False
                    break

            if append:
                points.append(point)
                goniometer_vals[vertical].append(horizontal)

    #for hor, vers in sorted(goniometer_vals.items(), key=lambda x: int(x[0])):
    #    print('{}: {}'.format(hor, vers))
    
    return np.array(points)




def test_rotate_vectors():
    '''

    '''
    
    points = []
    vectors = []
    
    horizontals = np.linspace(-60, 60, 10)
    verticals = np.linspace(0, 180, 10)

    for horizontal in horizontals:
        for vertical in verticals:
            point = camera2Fly(horizontal, vertical)
            vector = np.array([0,0,.2])
            
            P2 = force_to_tplane(point, point+vector)
            
            points.append(point)
            vectors.append(P2-point)
    
    points = np.array(points)
    vectors = np.array(vectors)

    points, vectors = rotate_vectors(points, vectors, radians(0), radians(89), 0)
    
    from gonioanalysis.drosom.plotting import vector_plot



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.view_init(elev=90, azim=0)
    vector_plot(ax, points, vectors)
    
    plt.savefig('test_rotate_vectors.png')


def test_imx():
    
    P0 = (2, -6)
    
    b = atan(P0[1] / P0[0])
    P1 = (P0[0]-sin(b), P0[1]+cos(b))


    plt.scatter(*P0, color='blue')
    plt.scatter(*P1, color='red')
    plt.scatter(0,0)

    plt.show()


def test_camvec2Fly():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(0,0,0, s=10) 
    
    horizontals = np.linspace(0, 360)

    for vertical in horizontals:

        imx, imy, horizontal = (1,0,0) 

        point0 = camera2Fly(horizontal, vertical)
        point1 = camvec2Fly(imx,imy,horizontal,vertical)

        print(point0)
        ax.scatter(*point0, s=10, color='blue')
        ax.scatter(*point1, s=10, color='red')
        



    plt.show()

def test1_camera_rotation():

    from drosom import get_data
    import tifffile
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from gonioimsoft.anglepairs import strToDegrees

    data = get_data('/home/joni/smallbrains-nas1/array1/pseudopupil_imaging/DrosoM23')

    for angle, image_fns in data.items():
        
        horizontal, vertical = strToDegrees(angle)
        rot = degrees(camera_rotation(horizontal, vertical))
        
        im = tifffile.imread(image_fns[0][0])
        im = ndimage.rotate(im, rot)
        
        print('{} {}'.format(horizontal, vertical))
        print(rot)

        plt.imshow(im)
        plt.show()


def test2_camera_rotation():

    horizontals = np.linspace(0,360)
    vertical = 0
    for horizontal in horizontals:
        rot = camera_rotation(horizontal, vertical)
        if rot != 0:
            raise ValueError('rot should be 0 for ALL horizontals when vertical = 0')
        
    horizontals = np.linspace(0,360)
    vertical = 180
    for horizontal in horizontals:
        rot = camera_rotation(horizontal, vertical)
        if round(degrees(rot)) != 180:
            raise ValueError('rot should be 180deg for ALL horizontals when vertical = -180. rot{}'.format(degrees(rot)))
    
    rot = camera_rotation(0, 95)
    if round(degrees(rot)) != 180:
        raise ValueError('rot should be 180deg for 0 horizontal when vertical = -95. rot{}'.format(degrees(rot)))
    
    rot = camera_rotation(-90, 45)
    if round(degrees(rot)) != 45:
        raise ValueError('rot should be 45deg for -90 horizontal when vertical = 45. rot{}'.format(degrees(rot)))
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    
    #test_camvec2Fly()
    #test_force_to_plane()
    #test1_camera_rotation()
    test_rotate_vectors()
