'''
Estimating optic flow field
'''

from math import cos, sin, radians

import numpy as np
from scipy.spatial import cKDTree as KDTree

import pupilanalysis.coordinates as coordinates
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

    P1 = coordinates.force_to_tplane(point, ov)
    
    P1 = coordinates.normalize(point, P1, scale=0.10)

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
    

    distances, indices = kdtree.query(points_A, k=10, n_jobs=-1)
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
        
        self.manalysers = [self]
        self.folder = 'optic_flow'
        self.eyes = ['left', 'right']
        self.vector_rotation = 0


        # FAnalyser specific
        self.pitch_rot = 0
        self.roll_rot = 0
        self.yaw_rot = 0
        self.points = {'right': coordinates.optimal_sampling(np.arange(0, 60, 5), np.arange(-100, 100, 5)),
                'left': coordinates.optimal_sampling(np.arange(-60, 0, 5), np.arange(-100, 100, 5))}
        
        self.constant_points = False

    def get_3d_vectors(self, eye, constant_points=None, *args, **kwargs):
        '''
        
        constant_points : bool
            If true, points stay the same and only vectors get rotated.
            If false, smooth rotation of the whole optic flow sphere.
        '''

        if constant_points is None:
            constant_points = self.constant_points

        if constant_points:
            # Rotate points, calculate vectors, rotate back
            points = coordinates.rotate_points(self.points[eye],
                    radians(self.yaw_rot),
                    radians(self.pitch_rot),
                    radians(self.roll_rot))
            
            points, vectors = coordinates.rotate_vectors(points, flow_vectors(points, xrot=0),
                    -radians(self.yaw_rot),
                    -radians(self.pitch_rot),
                    -radians(self.roll_rot))
        else:
            points = coordinates.optimal_sampling(np.arange(-90,90,5), np.arange(-180,180,5))
            points, vectors = coordinates.rotate_vectors(points, flow_vectors(points, xrot=0),
                    -radians(self.yaw_rot),
                    -radians(self.pitch_rot),
                    -radians(self.roll_rot))
            
            # Fixme. Make me with numpy, not list comprehension
            if eye == 'left':
                indices = [i for i, point in enumerate(points) if point[0] <= 0]
            elif eye == 'right':
                indices = [i for i, point in enumerate(points) if point[0] >= 0]

            points = [points[i] for i in indices]
            vectors = [vectors[i] for i in indices]
        
        return points, vectors

    
    def is_measured(self, *args, **kwargs):
        return True

    def are_rois_selected(self, *args, **kwargs):
        return True

    def load_analysed_movements(self, *args, **kwargs):
        return None

