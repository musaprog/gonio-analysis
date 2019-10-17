'''
Preprocessing pseudopupil imaging data to calculate binocular overlap.

Line scans assumed.

TODO
    BUG FIXES
    - cropping more than once
    - minimap size changes if image resolution changed and then contrast is changed by z or x

    CHANGES
    - calculate the cropping mean image using only corner images?

    FEATURES
    - Progress bar or wait text for slower operations such as loading full resolution images



'''

import time
from math import sqrt
import json

import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.widgets import RectangleSelector
import numpy as np

from drosox import XLoader
from images import ImageHandler, getFns

class RowAligner:
    '''
    Aligning stacks row-wise by user input.
    (Even though the step size is fixed, stacks at different pitch angles may
    not start from the same horizontal angle)
    '''

    def __init__(self, ImageHandler):
        
        if ImageHandler:
            
            self.handler = ImageHandler
            
            self.current = [0, 0]
            self.selection_mode = None
            self.shift_pressed = False
            self.quit = False

            # Information about how many images in each stack etc.
            self.N_images_list = [self.handler.getNImages(j) for j in range(self.handler.getNStacks())]
            
            # aligment_data
            self.centers = [False]*self.handler.getNStacks()
            self.row_shifts = [0]*self.handler.getNStacks()
            self.rotations = [0]*self.handler.getNStacks()
            self.crop = None
            self.both_visible = [[False for i in range(self.handler.getNImages(j))] for j in range(self.handler.getNStacks())]


            self.image_maxval = np.max(self.handler.getImage(*self.current)) / 2**3

            self.first_draw = True
            self.contrast_change = False

            self.minimap_patches = [[[] for i in range(self.handler.getNImages(j)) ] for j in range(self.handler.getNStacks())]
       


    def __autoRowShift(self):
        '''
        Automatically do row shifts based on marked centers.
        '''
        for j, center in enumerate(self.centers):
            if center:
                self.row_shifts[j] = - self.centers[j]

        # Push everything
        self.row_shifts = list( np.asarray(self.row_shifts) - np.min(self.row_shifts) )

    
    def __markCenter(self):
        self.centers[self.current[1]] = self.current[0] + self.row_shifts[self.current[1]]

   
    def __drawMiniMapOLD(self):
        '''
        Draws a minimap on the top left corner.
        '''
        #pixel_size = self.handler.getImageShape()[0]/100
        pixel_size=1

        for j in range(self.handler.getNStacks()):
            for i in range(self.handler.getNImages(j)):
                
                # Color of a minimap square
                # - Selection is always blue
                # - Rest of the row(s) red if center not marked
                # - After center set, the center is marked with yellow and other with gray
                # - if both pupils visible, color white
                if [i-self.row_shifts[self.current[1]], j] == self.current:
                    color = 'blue'
                else:
                    if not self.centers[j]:
                        color = 'red'
                    else:
                        if i == self.centers[j]:
                            color = 'yellow'
                        else:
                            
                            if self.both_visible[j][i]:
                                color = 'green'
                            else:
                                color = 'gray'
                
                xy = [(i+self.row_shifts[j])*pixel_size, j*pixel_size]
                
                # Keep minimap fully visible
                xy[0] -= np.min(self.row_shifts) * pixel_size

                if self.first_draw:
                    patch = matplotlib.patches.Rectangle(xy, pixel_size, pixel_size, edgecolor='black', facecolor=color)
                    self.minimap_patches[j][i] = patch
                    self.ax[0].add_patch(patch)
                else:
                    self.minimap_patches[j][i].set_facecolor(color)
                    self.minimap_patches[j][i].set_xy(xy)

        self.ax[0].set_xlim(np.min(self.row_shifts)*pixel_size, (np.max(self.row_shifts)+5+self.handler.getNImages(0))*pixel_size)
        self.ax[0].set_ylim(0, pixel_size*(self.handler.getNStacks()+1))
 
        #self.ax[0].autoscale(enable=True, tight=False)
    def __drawMiniMap(self):
        '''
        Draws a minimap on the top left corner.
        '''
        #pixel_size = self.handler.getImageShape()[0]/100
        

        M = self.handler.getNStacks()
        N = np.max([abs(self.handler.getNImages(j)+self.row_shifts[j]) for j in range(M)])
        
        minimap = np.zeros((N, M))

        minimap[self.current[1]][self.current[0]] = 1
        
        for j, i in enumerate(self.centers):
            if i:
                minimap[j][i] = 0.5
        
        self.ax[0].imshow(minimap)

        #self.ax[0].autoscale(enable=True, tight=False)
    
    def __updateImage(self):
        
       
        if self.selection_mode == 'crop':
            image= self.handler.getMeanImage()
        else:
            image = self.handler.getImage(self.current[0] + self.row_shifts[self.current[1]], self.current[1])
        
        image = np.clip(image, 0, self.image_maxval)
        

       


        if self.first_draw or self.contrast_change:
            self.imshow_obj =plt.imshow(image, cmap='gist_gray', interpolation=None)
            self.__drawMiniMap()
            self.first_draw = False
            self.contrast_change = False
            plt.show()
        else:
            self.imshow_obj.set_data(image)
            plt.draw()
        print('updating')
        
        self.__drawMiniMap()
        


    def __buttonPressed(self, event):
        '''
        A callback function connecting to matplotlib's event manager.
        '''

        if not self.shift_pressed:

            # Navigating between the images
            if event.key == 'down':

                self.current[0] -= 2*(self.row_shifts[self.current[1]-1] - self.row_shifts[self.current[1]])
                self.current[1] += -1

            elif event.key == 'up':
                
                self.current[0] -= 2*(self.row_shifts[self.current[1]+1] - self.row_shifts[self.current[1]])
                self.current[1] += 1

            elif event.key == 'left':
                self.current[0] += -1
            elif event.key == 'right':
                self.current[0] += 1
            
            # Image manipulations
            elif event.key == 'z':
                self.image_maxval /= sqrt(2)
                self.contrast_change = True
            elif event.key == 'x':
                self.image_maxval *= sqrt(2)
                self.contrast_change = True
            elif event.key == 'a':
                self.rectangle.set_active(True)
                self.selection_mode = 'align'
            
            # Making aligning
            elif event.key == 'm':
                self.__markCenter()
            elif event.key == 'y':
                self.both_visible[self.current[1]][self.current[0]+self.row_shifts[self.current[1]]] = True
            elif event.key == 'n':
                self.both_visible[self.current[1]][self.current[0]+self.row_shifts[self.current[1]]] = False


            # Load/save state
            elif event.key == 'w':
                self.saveAlign()
            elif event.key == 'b':
                self.loadBinarySearched()
        


        
        elif self.shift_pressed:
            if event.key == 'left':
                self.row_shifts[self.current[1]] += -1
                self.current[0] += +1
            
            elif event.key == 'right':
                self.row_shifts[self.current[1]] += 1
                self.current[0] += -1
            
            elif event.key == 'a':
                self.__autoRowShift()

            elif event.key in ['1','2','3','4','5','6']:
                self.handler.reloadImages(int(str(event.key)))
       

            self.shift_pressed = False
             
        if event.key == 'shift':
            self.shift_pressed = True
        
        self.__updateImage()

       
    def __onSelectRectangle(self, eclick, erelease):
        
        # Get selection box coordinates and set the box inactive
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rectangle.set_active(False)

        self.selection_mode = False

        self.__updateImage()


    def run(self, figure, ax):
        '''
        Runs the GUI at the given figure.
        '''

        self.figure = figure
        self.ax = ax
        self.cid = self.figure.canvas.mpl_connect('key_press_event', self.__buttonPressed)
          
        # Rectangle for cropping
        self.rectangle = RectangleSelector(ax[1], self.__onSelectRectangle, useblit=True)
        self.rectangle.set_active(False)

        self.ax[0].set_axis_off()
        self.ax[1].set_axis_off()



        self.__updateImage()


    def saveAlign(self):
        
        fn = 'row_shift.json'
        with open(fn, 'w') as fp:
            aligment_data = {'row_shifts': self.row_shifts,
                    'centers': self.centers,
                    'rotations': self.rotations,
                    'crop': self.crop,
                    'N_images_list': self.N_images_list,
                    'both_visible': self.both_visible}
            json.dump(aligment_data, fp)

    def loadAlign(self):
        fn = 'row_shift.json'
        with open(fn, 'r') as fp:
            aligment_data = json.load(fp)
            self.N_images_list = aligment_data['N_images_list']
            self.centers = aligment_data['centers']
            self.row_shifts = aligment_data['row_shifts']
            self.rotations = aligment_data['rotations']
            self.crop = aligment_data['crop']
            self.both_visible = aligment_data['both_visible']

    def loadBinarySearched(self):
        fn = 'binary_search_results/results.json'
        with open(fn, 'r') as fp:
            aligment_data = json.load(fp)
            self.centers = [aligment_data[i]['index_middle'] for i in range(self.centers)]
            
            #self.row_shifts = aligment_data['row_shifts']
            #self.both_visible = aligment_data['both_visible']

            #self.N_images_list = aligment_data['N_images_list']

def main():
    
    #path = '/work1/data/pseudopupils/drosoX2/'
    folder = '/win2/DrosoX5/'

    
    xloader = XLoader()
    data = xloader.getData(folder)
    handler = ImageHandler(data)



    fig, ax = plt.subplots(nrows=1, ncols=2)
    aligner = RowAligner(handler)
    aligner.run( fig, ax )
    


if __name__ == "__main__":
    main()
