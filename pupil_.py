
import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt

from pupil_imsoft.imsoft import loadAnglePairs, toDegrees
from pupil_detection import detect
from pupil_detection.training.import_positives import recursiveList



def pupilDetectImages(image_fns):
    '''
    Perform pupil detection to the given images.
    '''
    detection_results = []

    detector = detect.getDefaultDetector()

    for i, fn in enumerate(image_fns):
        print("Detecting image {}/{}".format(i+1, len(image_fns)))
        image = detect.openAndPreprocess(fn)
        head, eyes, pupils = detector.detect(image)
        
        detection_results.append((head,eyes,pupils))
        print(len(pupils))
        

    return detection_results


def processFolder():
    
    
    fns = recursiveList('/win2/DrosoX3/rot1sep/MMStack_Pos0.ome.tif', '.tif')
    fns.sort()
    angles = loadAnglePairs('/win2/DrosoX3/results_rot1.csv') 
    toDegrees(angles)
    
    fns = fns[0:len(angles)]

    X = [angle[0] for angle in angles]
    Y = [angle[1] for angle in angles]
            
    detection_results = pupilDetectImages(fns)
    
    # For now let's be interested just on how many pseudopupils are in a image
    pupils = np.asarray([len(x[2]) for x in detection_results])
    
    print(len(X))
    print(len(Y))
    print(len(pupils))
                 
    xi = np.linspace(np.min(X), np.max(X), 50)
    yi = np.linspace(np.min(Y), np.max(Y), 50)
    
    print(pupils) 
    zi = scipy.interpolate.griddata((X, Y), pupils, (xi[None,:], yi[:,None]), fill_value=0, method='nearest')
    
    #xi, yi = np.meshgrid(xi, yi)
    
    print(xi.shape)
    print(yi.shape)
    print(zi.shape)

    plt.pcolormesh(xi, yi, zi)
    plt.show()



def main():
    processFolder()


if __name__ == "__main__":
    main()



