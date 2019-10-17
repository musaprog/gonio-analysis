'''
Further analysis of the measured deep pseudopupil movement values.
'''

import numpy as np

from .optic_flow import flow_direction, field_error
from coordinates import nearest_neighbour


def optic_flow_error(manalyser, rotations):
    '''
    Calculates error between the measured pseudopupil movement and the theoretical
    optic flow for each measured point with theoretical optic flow coming from angles
    defined by rotations.

    Returns
        joined_points       Just the points going
    '''

    all_errors = []
    
    # Read 3D vectors beforehand so they don't have to be read many times
    points = {}
    vectors = {}
    
    for eye in ['left', 'right']:
        pointss, vectorss = manalyser.get_3d_vectors(eye)
        
        points[eye] = pointss
        vectors[eye] = vectorss
   

    # Calculate errors for each rotation
    for rot in rotations:

        errors = []

        for eye in ['left', 'right']:
            flow_vectors = [flow_direction(P0, xrot=rot) for P0 in points[eye]]
            errors.extend( field_error(vectors[eye], flow_vectors) )
        
        
        all_errors.append(errors)
        

    return np.concatenate((points['left'], points['right'])), all_errors
