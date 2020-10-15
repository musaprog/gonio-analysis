'''
Plot the results from spatial 2D calibration where the pinhole/fibre was
moved in the camera's (x, y) coordinates while only the green stimulus
LED was turned on and the fibre was connected to the spectrometer.
'''

import tifffile
import numpy as np
import matplotlib.pyplot as plt

from movemeter import Movemeter
from marker import Marker

def get_xy_coordinates(image_fn, match_image):
    '''
    Takes in an image of the pinhole and returs coordinates
    of the pinhole.
    
    image_fn        Image where we look for the match
    match_image     Matching target
    '''
    
    image = tifffile.imread(image_fn)

    movemeter = Movemeter()
    movemeter.set_data(image, )
    #plt.imshow(np.clip(image, np.min(image), np.percentile(image, 50)))
    #plt.show()



def main():
    image_fn = '/home/joni/smallbrains-nas1/array1/pseudopupil_joni/Spectrometer/DPP_cal_1_ty2/snap_2020-02-21_14.15.08.088000_0.tiff'
    
    fig, ax = plt.subplots()
    marker = Marker(fig, ax, [image_fn], None)
    pinhole = marker.run()    
    
    crop = pinhole[image_fn])
    pinhole_image = image_fn


    coodinates = get_xy_coordinates(image_fn)

    

if __name__ == "__main__":
    main()



