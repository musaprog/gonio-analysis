'''
Estimating optic flow field
'''

from math import cos, sin, radians

import numpy as np
from scipy.spatial import cKDTree as KDTree

from pupilanalysis.coordinates import force_to_tplane, normalize, optimal_sampling
from pupilanalysis.drosom.analysing import MAnalyser


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
    
    P1 = normalize(point, P1, scale=0.10)

    return P1-np.array(point)



def flow_vectors(points, xrot=0):
    '''
    Returns optic flow vectors (from flow_direction) as a numpy array.
    '''
    return np.array([flow_direction(P0, xrot=xrot) for P0 in points])


def field_error(points_A, vectors_A, points_B, vectors_B, direction=False, colinear=False):
    '''
    Relieved version where points dont have to overlap
    
    Put to A the eye vecotrs

    vectors_X   list of vector
    vector      (x,y,z)

    direction   Try to get the direction also (neg/pos)

    colinear : bool
    
    Returns the errors at points_A
    '''
    
    N_vectors = len(vectors_A)

    errors = np.empty(N_vectors)

    kdtree = KDTree(points_B)
    

    distances, indices = kdtree.query(points_A, k=10, n_jobs=2)
    weights = 1/(np.array(distances)**2)
    
    # Check for any inf
    for i_weights in range(weights.shape[0]):
        if any(np.isinf(weights[i_weights])):
            weights[i_weights] = np.isinf(weights[i_weights]).astype('int')

    #if any(np.isinf(weights):
    compare_vectors = [[vectors_B[i] for i in indx] for indx in indices]

    for i, (vecA, vecBs, vecB_weights) in enumerate(zip(vectors_A, compare_vectors, weights)):
        
        vec_errors = []
        
        for vecB in vecBs:

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

            vec_errors.append(error)

        errors[i] = np.average(vec_errors, weights=vecB_weights)

    if colinear:
        errors = 2 * np.abs(errors - 0.5)
    else:
        errors = 1 - errors

    return errors


class FAnalyser(MAnalyser):
    '''
    Sham analyser to just output optic flow vectors with the same
    api as MAnalyer does.
    '''
    
    def __init__(self, *args, **kwargs):
            
        print('inited')
        
        self.folder = 'optic_flow'
        self.eyes = ['left', 'right']
        self.vector_rotation = 0

        # FAnalyser specific
        self.xrot = 0
        self.points = {'left': optimal_sampling(np.arange(0, 50, 5), np.arange(-100, 100, 5)),
                'right': optimal_sampling(np.arange(-50, 0, 5), np.arange(-100, 100, 5))}


    def get_3d_vectors(self, eye, *args, **kwargs):
        
        vectors = flow_vectors(self.points[eye], xrot=self.xrot)
        return self.points[eye], vectors

    
    def is_measured(self, *args, **kwargs):
        return True

    def are_rois_selected(self, *args, **kwargs):
        return True

    def load_analysed_movements(self, *args, **kwargs):
        return None

