'''
For antenna level search of DrosoM
'''

import os


def loadReferenceFly(folder):
    '''
    Returns the reference fly data, dictionary with pitch angles as keys and
    image filenames as items.
    '''
    pitches = []
    with open(os.path.join(folder, 'pitch_angles.txt'), 'r') as fp:
        for line in fp:
            pitches.append(line)

    images = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('.tif') or fn.endswith('.tiff')]
    images.sort() 
    return {pitch: fn for pitch,fn in zip(pitches, images)}


