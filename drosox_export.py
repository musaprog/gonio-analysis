'''
Export DrosoX images.
'''

import os.path
import sys

import csv

import cv2
import tifffile

from drosox import XLoader
from image_adjusting import ImageAdjuster


class XExporter:

    def __init__(self):
        self.loader = XLoader()
        self.adjuster = ImageAdjuster()


    def export(self, data_folder, target_folder):
        data = self.loader.getData(data_folder)
        
        i,j = [0,0]
        angles_and_filenames = []
        
        z=0
        for pitch, horizontals_and_fns in data:
        
       

            #self.adjuster.determineROI([z[1] for z in horizontals_and_fns])

            for horizontal, fn in horizontals_and_fns:
               
                image = tifffile.imread(fn) 
                #image = self.adjuster.autoAdjust(image)

                savefn =  os.path.join(target_folder, "im_{0:0>6}.tif".format(z))
                short_savefn = os.path.split(savefn)[1]
                print("Writing {} to {}".format(fn, savefn))
                cv2.imwrite(savefn, image)

                angles_and_filenames.append([short_savefn, pitch, horizontal])

                z+=1
            
        with open(os.path.join(target_folder, 'angles_and_filenames.csv'), 'w') as fp:
            writer = csv.writer(fp)
            for fn, pitch, horizontal in angles_and_filenames:
                writer.writerow([fn, horizontal, pitch])

def main():
    '''
    Usage from the command line:
    argv1    Path to the DrosoX data
    argv2    Target directory
    '''
    drosox_data, export_target = sys.argv[1:3]

    exporter = XExporter()
    exporter.export(drosox_data, export_target)
    

if __name__ == '__main__':
    main()
