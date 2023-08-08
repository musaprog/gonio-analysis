'''
Estimating optic flow field
'''

from math import cos, sin, radians

import numpy as np
import scipy
from scipy.spatial import cKDTree as KDTree
from scipy.stats import mannwhitneyu

import gonioanalysis.coordinates as coordinates
from gonioanalysis.drosom.analysing import MAnalyser
from gonioanalysis.version import used_scipy_version


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
    
    if used_scipy_version < (1,6,0):
        distances, indices = kdtree.query(points_A, k=10, n_jobs=-1)
    else:
        distances, indices = kdtree.query(points_A, k=10, workers=-1)
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
            
            if direction:
                counter = coordinates.rotate_along_arbitrary(points_A[i], vecB, angle) 
                clock = coordinates.rotate_along_arbitrary(points_A[i], vecB, -angle)
                
                if np.sum(counter - vecB) > np.sum(clock - vecB):
                    error = -error

            vec_errors.append(error)

        errors[i] = np.average(vec_errors, weights=vecB_weights)
    
    if direction:
        if colinear:
            errors *= 2
        errors = (errors + 1)/2
    else:
        if colinear:
            errors = 2 * np.abs(errors - 0.5)
        else:
            errors = 1 - errors

    return errors


def _find_unique_points(points):
    '''
    Returns
    -------
    points : list
    indices : dict of int
    '''

    unique_points = list({p for p in points})
    indices = [[] for i in range(unique_points)]
    for i_point, point in enumerate(points):
        indices[unique_points.index(point)].append(i_point)
    
    return unique_points, indices


def _angle(self, vecA, vecB):
    return np.arccos(np.inner(vecA, vecB)/(np.linalg.norm(vecA) * np.linalg.norm(vecB)))

def field_pvals(points_A, vectors_A, points_B, vectors_B, direction=False, colinear=False):
    '''
    Assuming 
    '''
    # same points are repeated many times (otherwise it would make sense
    # to do statistical testing. Find the these "unique" points
    un_points_A, un_indices_A = _find_unique_points(points_A)
    un_points_B, un_indices_B = _find_unique_points(points_B)

    kdtree = KDTree(points_B)
    
    for point, indices_A in zip(un_points_A, un_indices_A):
        # Closest point
        if used_scipy_version < (1,6,0):
            distance_B, index_B = kdtree.query(point, k=1, n_jobs=-1)
        else:
            distance_B, index_B = kdtree.query(point, k=1, workers=-1)
        
        Avecs = [vectors_A[i] for i in indices_A]
        Bvecs = [vectors_B[i] for i in un_indices_B[index_B]]
        
        # Mean of Avecs
        mean_Avec = np.mean(Avecs, axis=0)
        
        # Relative rotations of vectors with respect to the mean Avec
        Adirs = [_angle(vec, mean_Avec) for vec in Avecs]
        Bdirs = [_angle(vec, mean_Avec) for vec in Avecs]

        u_stats, pval = mannwhitneyu(Adirs, Bdirs)
        
        pvals.append(pval)

    return un_points_A, pvals



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
        

        self.ui_options = {
                'pitch_rot': {'help': 'Pitch rotation', 'type': float},
                'roll_rot': {'help': 'Roll rotation', 'type': float},
                'yaw_rot': {'help': 'Yaw rotation', 'type': float},
                }


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
        
        return np.array(points), np.array(vectors)

    
    def is_measured(self, *args, **kwargs):
        return True

    def are_rois_selected(self, *args, **kwargs):
        return True

    def load_analysed_movements(self, *args, **kwargs):
        return None


