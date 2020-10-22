'''
For antenna level search of DrosoM
'''

import os

from pupilanalysis.directories import CODE_ROOTDIR


def loadReferenceFly(folder):
    '''
    Returns the reference fly data, dictionary with pitch angles as keys and
    image filenames as items.
    '''
    pitches = []
    root_dir = os.path.dirname
    with open(os.path.join(CODE_ROOTDIR, folder, 'pitch_angles.txt'), 'r') as fp:
        for line in fp:
            pitches.append(line)

    images = [os.path.join(CODE_ROOTDIR, folder, fn) for fn in os.listdir(os.path.join(CODE_ROOTDIR, folder)) if fn.endswith('.tif') or fn.endswith('.tiff')]
    images.sort() 
    return {pitch: fn for pitch,fn in zip(pitches, images)}


