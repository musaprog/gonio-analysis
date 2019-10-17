'''
Adjusting behind-illuminated pseudopupil images
'''

import os
import itertools
import multiprocessing
import multiprocessing.dummy

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pupil_detection import detect

import tifffile


class ImageAdjuster:

    def __init__(self):
        self.detector = detect.getDefaultDetector()

    def determineROI(self, fns):
        ROIs = []
        for image in fns:
            ROI = self.detector.detectHead(image)
            if ROI:
                ROIs.append(ROI)

        if ROIs:
            self.ROI = [int(z) for z in np.mean(ROIs, axis=0)[0]]


    def autoAdjust(self, image):
        '''
        Tries to automatically adjust a image by first detectin fly heads and using
        the average of these regions to autoadjust contrast.

        Works best when images are not too different from each other.
        '''
        
        (x,y,w,h) = self.ROI

        im_ROI = image[y:y+h, x:x+w]
        #r, image = cv2.threshold(image, np.max(im_ROI), 0, 2) 
        image = np.clip(image, np.min(im_ROI), np.mean(im_ROI)*2)
        normalized = np.zeros(image.shape)
        image = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        
        return image



class GeneralAdjuster:
    '''
    Parent class for other adjusters, implements
    saving to disk etc.
    '''

    def adjust():
        raise NotImplementedError



class ROIAdjuster(GeneralAdjuster):
    '''
    Use provided ROI to perform the adjusting.
    '''
    
    def adjust(self, image, ROI, extend_factor=1):
        '''
        Returns adjusted image.
        
        INPUT ARGUMENTS     DESCRIPTION
        ROI                 (x,y,w,h)          
        extend_factor       Scales ROI larger (>1) or smalle (<1)
        '''
        
        ROI = [int(round(i)) for i in ROI]
        (x,y,w,h) = ROI
        
        

        black = 80
        white = 20
        

        im_ROI = image[y:y+h, x:x+w]
               
        x += w - int(w*extend_factor/2)
        y += h - int(h*extend_factor/2)
        w = w*extend_factor
        h = h*extend_factor

        im_ROI_extended = image[y:y+h, x:x+w]
        


        clip_points = [np.percentile(im_ROI_extended, 1), np.percentile(im_ROI, 99)]
        clip_points[0] -= ( (clip_points[0]+clip_points[1])/2 ) / 3.14
        clip_points[1] += ( (clip_points[0]+clip_points[1])/2 ) / 3.14
            

        image = np.clip(image, *clip_points)


        normalized = np.zeros(image.shape)
        image = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        
        cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0,), 3)

        return image

    
    def singleWriteAdjusted(self, fns, ROIs, new_fns, extend_factor=1, binning=1):
        '''
        Single threaded version
        '''
        writed_fns = []
        
        for i in range(0, len(fns), binning):
            print('{} to {}'.format(fns[i], new_fns[i]))
            os.makedirs(os.path.dirname(fns[i]), exist_ok=True)
            
            image = self.adjust(tifffile.imread(fns[i]), ROIs[i], extend_factor=extend_factor) / binning
            for j in range(1, binning):
                try:
                    image += self.adjust(tifffile.imread(fns[i+1]), ROIs[i+1], extend_factor=extend_factor) / binning
                except IndexError:
                    image = image * (binning/j)
                    break

            cv2.imwrite(new_fns[i], image)
            writed_fns.append(new_fns[i])

        return writed_fns


    def _wrapWriteAdjusted(self, input_list):
        args, key_args = input_list
        return self.singleWriteAdjusted(*args, extend_factor=key_args[0], binning=key_args[1])


    def writeAdjusted(self, fns, ROIs, new_fns, extend_factor=1, binning=1):
        '''
        Writes adjusted images to disk.
        '''
        threads = multiprocessing.cpu_count() * 2
        work_size = int(len(fns)/threads)
        
        works = []

        for i in range(0, threads):
            works.append([])
            
            a = i*work_size

            if i == threads-1:
                b = len(fns)-1
            else:
                b = (i+1)*work_size
            
            works[-1].append([fns[a:b], ROIs[a:b], new_fns[a:b]])
            works[-1].append([extend_factor, binning])

        
        with multiprocessing.Pool(threads) as p:
            writed_fns = list(itertools.chain(*list(p.map(self._wrapWriteAdjusted, works))))
        # 
        # FIXME writed_fns is an empty list
        
        print('Writed fns {}'.format(writed_fns))

        return writed_fns
        #return list(itertools.chain(*writed_fns))
        #for fn, ROI, new_fn in zip(fns, ROIs, new_fns):
        #    print('{} to {}'.format(fn, new_fn))
        #    os.makedirs(os.path.dirname(fn), exist_ok=True)
        #    image = self.adjust(tifffile.imread(fn), ROI, extend_factor=extend_factor)
        #    cv2.imwrite(new_fn, image)
    





def main():
    
    import os.path

    path = "/win2/DrosoX6/rot"
    fns = ['img_channel000_position000_time000004353_z000.tif', 'img_channel000_position000_time000009107_z000.tif',
            'img_channel000_position000_time000013861_z000.tif', 'img_channel000_position000_time000018615_z000.tif',
            'img_channel000_position000_time000004354_z000.tif', 'img_channel000_position000_time000009108_z000.tif']
    
    fns = [os.path.join(path,fn) for fn in fns]

    images = [tifffile.imread(fn) for fn in fns]


    adjuster = ImageAdjuster()
    adjuster.autoAdjust(fns, images) 




if __name__ == "__main__":
    main()
