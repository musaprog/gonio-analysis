'''
Further analysis of the measured deep pseudopupil movement values.
'''

import numpy as np

from .optic_flow import flow_direction, field_error
from pupil.coordinates import nearest_neighbour, force_to_tplane




def optic_flow_error(manalyser, rotations, self_error=False):
    '''
    Calculates error between the measured pseudopupil movement and the theoretical
    optic flow for each measured point with theoretical optic flow coming from angles
    defined by rotations.

    Returns
        left and right points           (hor,ver) points for left and right eyes,
                                        where the errors where calculated
        all_errors                      List of errors for each fly rotation

    FIXME What is self_error?
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
            
            if self_error:
                reference_vectors = [force_to_tplane(point, np.array([1,0,0])) for point in points[eye]]
                errors.extend( field_error(vectors[eye], reference_vectors))
                continue

            flow_vectors = [flow_direction(P0, xrot=rot) for P0 in points[eye]]
            errors.extend( field_error(vectors[eye], flow_vectors) )
        
        
        all_errors.append(errors)
        

    return np.concatenate((points['left'], points['right'])), all_errors
