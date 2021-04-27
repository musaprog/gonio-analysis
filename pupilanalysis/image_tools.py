
import numpy as np
import tifffile
import matplotlib.pyplot as plt


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



class ImageShower:
    '''
    From  cascade training, a generalized image shower
    '''
    
    def __init__(self, fig, ax):
        
        self.fig = fig
        self.ax = ax
        
        self.fns = None

        self.buttonActions = []
        self.image_brightness = 0
        self.image_maxval = 1
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.callbackButtonPressed)
        
        self.pupils = -1
        
        self.title = ''

        self.fig.show()


    def callbackButtonPressed(self, event):
        '''
        A callback function connecting to matplotlib's event manager.
        '''
        
        # Navigating between the images
        if event.key == 'z':
            self.image_maxval -= 0.3
            self.updateImage(strong=True)
        
        elif event.key == 'x':
            self.image_maxval += 0.3
            self.updateImage(strong=True)
        
        elif event.key == 'a':
            self.image_brightness += 50
            self.updateImage(strong=True)
        elif event.key == 'c':
            self.image_brightness += -50
            self.updateImage(strong=True)
        
        for button, action in self.buttonActions:
            if event.key == button:
                print(event.key)
                action()


    def setImages(self, image_fns):
        '''
        Set the images that ImageShower shows.
        '''
        self.fns = image_fns
        self.cache = {}

    def setImage(self, i):
        fn = self.fns[i]
        try:
            self.image = self.cache[fn]
        except KeyError:
            self.image = tifffile.imread(fn) 
        
        self.updateImage()
        
    def cacheImage(self, i):
        '''
        Loads image to cache for faster showup. 
        '''
        fn = self.fns[i]
        # Uncomment to take cahcing in use
        #self.cache[fn] = tifffile.imread(fn)

    def setTitle(self, title):
        self.title = title

    def updateImage(self, strong=False):
        capvals = (0, np.mean(self.image) *self.image_maxval)
        self.ax.clear() 
        self.ax.set_title(self.title)
        self.ax.imshow(self.image-self.image_brightness,cmap='gist_gray', interpolation='nearest', vmin=capvals[0], vmax=capvals[1])
        
        if not strong:
            self.fig.canvas.draw()
        else:
            self.fig.show()
    
    def close(self):
        plt.close(self.fig)


