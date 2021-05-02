import numpy as np

from gonioanalysis.coordinates import force_to_tplane, normalize

def get_reference_vector(P0):
    '''
    Generate a reference vector for the coffiecient of determination
    (R**2 value) calculation.

    The reference vectors are always in the XY-plane and the sphere's
    tangential plane pointing east.
    
    Returns vec so that P0+vec=P1
    '''
    px,py,pz = P0
    aP0 = np.array(P0)
    
    if px==0:
        if py > 0:
            phi = np.pi/2
        else:
            phi = -np.pi/2

    else:
        phi = np.arctan(py/px)
    
    if px < 0:
        phi += np.pi

    # x,y,z components
    vx = -np.sin(phi)
    vy = np.cos(phi)
    
    vec = force_to_tplane(aP0, aP0+np.array((vx,vy,0)))
    vec = normalize(aP0, vec, scale=.15)
    vec = vec - aP0

    return vec
    
        


