
import numpy as np


def mean_max_response(manalyser, image_folder):
    '''
    Averages over repetitions, and returns the maximum of the
    mean trace.

    manalyser
    image_folder
    '''
    
    displacements = manalyser.get_displacements_from_folder(image_folder)
    
    return np.max(np.mean(displacements, axis=0))


