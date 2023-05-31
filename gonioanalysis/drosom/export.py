'''Export data from MAnalyser objects
'''

import math
import datetime
import json

import numpy as np

from gonioanalysis.coordinates import get_rotation_matrix
from .optic_flow import field_error

# Available export filetypes
FILETYPES = [('JSON', '.json'), ('Numpy binary', '.npy')]


def vectors_to_yxz_rotations(points, vectors):
    '''Converts 3D vectors to yxz rotations for external applications.

    Excexution: Have an arrow 3D model at origo, pointing towards positive X.
    Then, rotate the arrow along Y axis (the arrow end still remaining at origo).
    Then, move the arrow to (0,1,0) and while having the center of rotation
    at origo (0,0,0), rotate first along the X axis and then along the Z axis.
    After these steps you have reconstructed the vector map.

    Arguments
    ---------
    points : list
        Vector locations from get_3d_vectors
    vectors : list
        Vectors themselves from get_3d_vectors

    Returns
    -------
    yxz_rots : list
        List of rotations [(yr, xr, zr), ...]
    '''
    
    ## Transform points and points+vectors in reverse to the Y-axis
    zrots = [math.atan2(y, x)-math.pi/2 for x,y,z in points]

    rotated_points = [get_rotation_matrix('z', -rot) @ point for rot, point in zip(zrots, points)]
    rotated_vectors = [get_rotation_matrix('z', -rot) @ vector for rot, vector in zip(zrots, vectors)]

    # points should be all now in in yz plane
    #xrots = [math.asin(z) for x,y,z in rotated_points]
    xrots = [math.atan2(z, y) for x,y,z in rotated_points]
    rotated_points = [get_rotation_matrix('x', -rot) @ point for rot, point in zip(xrots, rotated_points)]
    rotated_vectors = [get_rotation_matrix('x', -rot) @ vector for rot, vector in zip(xrots, rotated_vectors)]

    # calculate rotation of the vector when point now on (0,1,0)
    yrots = [math.atan2(-x, z) - math.pi for x,y,z in rotated_vectors]
    
    return [[yr,xr,zr] for yr,xr,zr in zip(yrots, xrots, zrots)]


def _date():
    return str(datetime.datetime.now())


def _export_data(data, save_fn):
    if save_fn.endswith('.json'):
        with open(save_fn, 'w') as fp:
            json.dump(data, fp)
    else:
        raise ValueError(f'Unsupported filetype: {save_fn}')

def export_vectormap(analyser, save_fn=None):
    '''Exports 3D vectors

    save_fn : string
        Path to the new file
    '''

    if save_fn is None:
        saven_fn = f'vectormap_{analyser.name}_{_date()}.npy'

    if save_fn.endswith('.npy'):
        base = save_fn.removesuffix('.npy')
        for eye in ['left', 'right']:
            vectors =  analyser.get_3d_vectors(eye)
            np.save(base+'_'+eye+'.npy', vectors)
    
    elif save_fn.endswith('.json'):
        data = {}
        for eye in ['left', 'right']:
            points, vectors = analyser.get_3d_vectors(eye)
            data[eye] = {
                    'points': points.tolist(), 
                    'vectors': vectors.tolist(),
                    'yxz_rotations': vectors_to_yxz_rotations(points, vectors)}
        
        with open(save_fn, 'w') as fp:
            json.dump(data, fp)

    else:
        raise ValueError(f'Unkown file ending: {save_fn}')




def export_differencemap(analyser1, analyser2, save_fn=None):
    '''Exports the comparision between two analysers' vectormaps.

    The differences (aka. errors) range between 0 and 1 and are
    in the points of the analyser1.
    '''
    
    data = {}
    
    for eye in analyser1.eyes:
        p1, v1 = analyser1.get_3d_vectors(eye)
        p2, v2 = analyser2.get_3d_vectors(eye)
        errors = field_error(p1, v1, p2, v2)
        
        print(len(p1))
        print(len(p2))

        data[eye] = {
                'points1': p1.tolist(),
                'points2': p2.tolist(),
                'vectors1': v1.tolist(),
                'vectors2': v2.tolist(),
                'yxz_rotations1': vectors_to_yxz_rotations(p1, v1),
                'yxz_rotations2': vectors_to_yxz_rotations(p2, v2),
                'errors': errors.tolist()
                }

    if save_fn is None:
        save_fn = f'differencemap_{analyser1.name}_{analyser2.name}_{_date()}.json'

    _export_data(data, save_fn)

