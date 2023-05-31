'''Export data from MAnalyser objects
'''

import math
import datetime
import json

import numpy as np

from gonioanalysis.coordinates import get_rotation_matrix


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



def export_vectormap(analyser, save_fn=None):
    '''Exports 3D vectors

    save_fn : string
        Path to the new file
    '''

    if save_fn is None:
        date = datetime.datetime.now()
        saven_fn = f'vectormap_{analyser.name}_{date}.npy'

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



