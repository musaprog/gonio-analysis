
import numpy as np
import tifffile

def open_adjusted(fn):
    '''
    Images can have high glares. This function tries to
    clip the image (adjust contrast) so that the DPP
    is clearly visible.

    Usually the glare is at one point so this uses
    the fact.
    '''

    image = tifffile.imread(fn)

    glare_max = np.max(image)
    median = np.median(image)

    image = np.clip(image, 0, int(np.mean([glare_max, median])/6) )

    return image

