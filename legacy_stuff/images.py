'''

Manipulating images mainly by ImageHandler class.

'''
import os
import json
import uuid
import tifffile

import numpy as np


IMAGE_CACHE_DIR = 'tmp/'


def getTestFns():
    return getFns('/work1/data/pseudopupils/drosoX1/')

def getFns(path):
    '''
    Returns tif files inside a drosoX folder.

    drosoX
        ...
        a_folder
            a_stack.tiff
        a_next_folder
            a_stack.tiff
        ...

    Naming of the files or folders does not matter.

    '''
    

    folders = [os.path.join(path, item) for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    
    stacks = []
    for folder in folders:
        stack = [os.path.join(folder, item) for item in os.listdir(folder) if 'tif' in item[-4:]]
        stacks.append(stack)
    
    stacks.sort()
    return stacks


class DiskCacheManager:
    '''
    Keeping record of cached files.
    Caching happens on filename basis.
    '''

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_fn = os.path.join(cache_dir, 'cache.csv')

        if os.path.exists(self.cache_fn):
            
            with open(self.cache_fn, 'r') as fp:
                self.cache = json.load(fp)
        else:
            self.cache = {}


    def __str(self, options):
        return str(options)

    def writeCachedFn(self, real_fn, options):
        '''
        Write a new cahced_fn with real_fn, options combo.
        '''
        cached_fn = str(uuid.uuid4()) + '.tif'
        
        try:
            self.cache[real_fn][self.__str(options)]
        except KeyError:
            self.cache[real_fn] = {}

        self.cache[real_fn][self.__str(options)] = cached_fn
        
        return os.path.join(self.cache_dir, cached_fn)

    def getCachedFn(self, real_fn, options):
        '''
        Get cached_fn if one exists corresponding to real_fn, options combo.
        Returns false if cached file not existing.
        '''
        try:
            cached_fn = self.cache[real_fn][self.__str(options)]
        except KeyError:
            return False

        return os.path.join(self.cache_dir, cached_fn)
    
    def saveCache(self):
        '''
        Save current cache database to the disk.
        '''
        with open(self.cache_fn, 'w') as fp:
            json.dump(self.cache, fp)



class ImageHandler():
    '''
    Handling images. 
    '''

    def __init__(self, groupedAnglesAndImages):
        '''
        stack_fns = []
        '''

        self.image_fns = []
        for pitch, horizontals in groupedAnglesAndImages:
            self.image_fns.append([])
            for (horizontal, fn) in horizontals:
                self.image_fns[-1].append(fn)
        
        self.loadedImages = {}

    def loadImage(self, fn, downscaler=1):

        cache_mg = DiskCacheManager(IMAGE_CACHE_DIR)
    
        cache_fn = cache_mg.getCachedFn(fn[0], downscaler)
        if cache_fn:
            image = tifffile.imread(cache_fn)   
        else:
            image = tifffile.imread(fn)
        
        self.loadedImages[fn] = image


    def getNStacks(self):
        return len(self.image_fns)

    def getNImages(self, j_stack):
        return len(self.image_fns[j_stack])


    def getImageShape(self):
        '''
        Returns the current dimensions of images in pixels.
        '''
        return self.getImage(0, 0).shape


    def rotateStack(self, j_stack):
        for i in range(self.stacks[j_stack].shape[0]):
            self.stacks[j_stack][i] = np.rot90( self.stacks[j_stack][i] )

    def getImage(self, i_image, j_stack):
        '''
        Returns the i:th image from the j:t stack.

        If cropped version by self.cropImage exsits, by default return this.
        '''

        fn = self.image_fns[j_stack][i_image]
        print(fn) 
        if not fn in self.loadedImages.keys():
            self.loadImage(fn)
        
        return self.loadedImages[fn]

    def getMeanImage(self):
        '''
        Returns the mean image of all images taking any croppings into account.
        '''
        images = []
        for j in range(self.getNStacks()):
            for i in range(self.getNImages(j)):
                images.append(self.getImage(i, j))

        return np.mean(images, axis=0)
     


def unit_test():
    
    import matplotlib.pyplot as plt
    
    fns = ['/work1/data/pseudopupils/drosoX1/fov_2.5steps_20deg/MMStack_Pos0.ome.tif']

    handler = ImageHandler(fns)
    im = handler.getImage(0, 0)
    plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    unit_test()
