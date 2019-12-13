'''
Estimating optic flow field
'''

from math import cos, sin, radians

import numpy as np

from pupil.coordinates import force_to_tplane, normalize

def flow_direction(point, xrot=0):
    '''
    Estimates optic flow at the given point by placing the optic flow vector to
    the point, and forcing it to the tangent plane of the sphere.

    The optic flow vector is a unit vector -j (ie. pointing to negative y-axis),
    unless rotated. This should be okay since we are interested only in the
    direction of the flow field, not magnitude.

    INPUT ARGUMENTS         DESCRIPTION
    point                   (x0,y0,z0)
    xrot                    Rotation about x-axis

    RETURNS
    xi,yj,zk    Flow direction vector at origo
    '''
    
    rxrot = radians(xrot)
    ov = 2*np.array(point) + np.array([0,-1*cos(rxrot),sin(rxrot)])

    P1 = force_to_tplane(point, ov)
    
    P1 = normalize(point, P1, scale=0.15)

    return P1-np.array(point)



def flow_vectors(points, xrot=0):
    '''
    Returns optic flow vectors (from flow_direction) as a numpy array.
    '''
    return np.array([flow_direction(P0, xrot=xrot) for P0 in points])



def field_error(vectors_A, vectors_B, direction=False):
    '''

    vectors_X   list of vector
    vector      (x,y,z)

    direction   Try to get the direction also (neg/pos)
    '''
    
    if len(vectors_A) != len(vectors_B):
        raise ValueError('vectors_A and vectors_B have to be same length')
    
    N_vectors = len(vectors_A)

    #mean_error = 0

    errors = np.empty(N_vectors)

    for i, (vecA, vecB) in enumerate(zip(vectors_A, vectors_B)):
    
        angle = np.arccos(np.inner(vecA, vecB)/(np.linalg.norm(vecA) * np.linalg.norm(vecB)))
        error = angle / np.pi
        if not 0<=error<=1:
            # Error is nan if either of the vectors is zero because this leads to division
            # by zero because np.linalg.norm(vec0) = 0
            # -> set error to 1 if vecA != vecB or 0 otherwise
            if np.array_equal(vecA, vecB):
                error = 0
            else:
                error = 1
        
        if direction and vecB[2] > vecA[2]:
            error = -error

        errors[i] = error
    #print('max error {} and min {}'.format(np.max(errors), np.min(errors)))
    return errors




