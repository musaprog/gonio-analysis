'''
Working in 3D space
'''
import math
from math import sin, cos, tan, radians, pi, acos, atan, sqrt, degrees, atan2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def camera_rotation(horizontal, vertical):
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
    print('imx is {}'.format(imx))
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
    from pupil_imsoft.anglepairs import strToDegrees

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
    #test_camvec2Fly()
    #test_force_to_plane()
    test1_camera_rotation()
