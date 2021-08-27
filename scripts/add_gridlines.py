'''
Add grid lines to the selected GHS-DPP images
'''
import os

import numpy as np
from tkinter import filedialog, simpledialog
import cv2

def main():

    fns = filedialog.askopenfilenames()
    
    if fns:

        print(fns)

        pixel_size = 0.817
        every = 20

        directory = os.path.dirname(fns[0])
        
        newdir = os.path.join(directory, 'gridded')
        os.makedirs(newdir, exist_ok=True)
        
        for fn in fns:
            image = cv2.imread(fn)
            

            for i_line, j in enumerate(np.arange(0, image.shape[0]-1, every/pixel_size)):
                j = int(j)
                
                if i_line == 13:    
                    image[j:j+1, :, 2] = 255
                else:
                    image[j:j+1, :, 0] = 255
            
            for i_line, i in enumerate(np.arange(0, image.shape[1]-1, every/pixel_size)):
                i = int(i)

                if i_line == 10: 
                    image[:, i:i+1, 2] = 255
                else:
                    image[:, i:i+1, 0] = 255

            cv2.imwrite(os.path.join(newdir, os.path.basename(fn)), image)


if __name__ == "__main__":
    main()


