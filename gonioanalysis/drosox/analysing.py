import os
import json
import time

import matplotlib.pyplot as plt

from gonioanalysis.directories import PROCESSING_TEMPDIR
from gonioanalysis.drosox.loading import load_data
from gonioanalysis.binary_search import (
        binary_search_middle,
        binary_search_left,
        binary_search_right
)
from gonioanalysis.image_tools import ImageShower



class XAnalyser:
    '''
    Analysing and getting the analysed results out
    '''
    def __init__(self, data_path, folder):
        '''
        data_path       Path to the data folders
        folder          Name of the DrosoX folder
        '''
        self.data_path = data_path
        self.folder = folder
        self.fly = folder
        
        # Move these away from here
        self.males = ['DrosoX6', 'DrosoX7', 'DrosoX8', 'DrosoX9', 'DrosoX15']
        self.females = ['DrosoX10', 'DrosoX11', 'DrosoX12', 'DrosoX13', 'DrosoX14']
        self.skip_flies = ['DrosoX14']
        
        # Set saving directories and create them
        self.savedirs = {'overlap': 'binocular_overlap', 'level': 'vertical_correction'}
        for key in self.savedirs:
            self.savedirs[key] = os.path.join(PROCESSING_TEMPDIR,
                    'XAnalyser_data',
                    self.savedirs[key])
            os.makedirs(self.savedirs[key], exist_ok=True)
    
        
        print('Initializing XAnalyser, datapath {}, folder {}'.format(data_path, folder))
    

    def get_data(self):
        '''
        Calls drosox.loading.load_data for this fly.
        '''
        return load_data(os.path.join(self.data_path, self.folder))
    

    def measure_overlap(self):
        '''
        Analyses binocular overlap by the binary search method, where
        the user makes decisions wheter the both pseudopupils are visible
        or not.
        '''
        start_time = time.time()
        
        data = load_data(os.path.join(self.data_path, self.folder))

        fig, ax = plt.subplots()
        shower = ImageShower(fig, ax)

        # Try to open if any previously analysed data
        analysed_data = []
        try: 
            with open(os.path.join(PROCESSING_TEMPDIR, 'binary_search', 'results_{}.json'.format(fly)), 'r') as fp:
                analysed_data = json.load(fp)
        except:
            pass
        analysed_pitches = [item['pitch'] for item in analysed_data]
       
        print('Found {} pitches of previously analysed data'.format(len(analysed_data)))

        for i, (pitch, hor_im) in enumerate(data):
            
            # Skip if this pitch is already analysed
            if pitch in analysed_pitches:
                continue

            horizontals = [x[0] for x in hor_im]
            images = [x[1] for x in hor_im]

            N = len(images)
            shower.setImages(images)

            # Ask user to determine middle, left, and right
            i_m = binary_search_middle(N, shower)
            if i_m == 'skip':
                continue

            i_l = binary_search_left(N, shower, i_m)
            i_r = binary_search_right(N, shower, i_m)
            
            analysed_data.append({})

            analysed_data[-1]['N_images'] = N
            analysed_data[-1]['pitch'] = pitch
            analysed_data[-1]['horizontals']= horizontals
            analysed_data[-1]['image_fns']= images
           
            analysed_data[-1]['index_middle'] = i_m
            analysed_data[-1]['index_left'] = i_l
            analysed_data[-1]['index_right']= i_r
            
            analysed_data[-1]['horizontal_middle'] = horizontals[i_m]
            analysed_data[-1]['horizontal_left'] = horizontals[i_l]
            analysed_data[-1]['horizontal_right']= horizontals[i_r]
            

            print('Done {}/{} in time {} minutes'.format(i+1, len(data), int((time.time()-start_time)/60) ))
            
            # Save on every round to avoid work loss
            with open(os.path.join(self.savedirs['overlap'],
                'results_{}.json'.format(self.fly)), 'w') as fp:
                json.dump(analysed_data, fp)


    
    def get_overlaps(self, correct_antenna_level=True):
        '''
        Load the results of binary search.
        
        correct_antenna_level       Corrects with antenna level
        '''
        fn = os.path.join(self.savedirs['overlap'], 'results_{}.json'.format(self.fly))
        
        
        with open(fn, 'r') as fp:
            overlap_markings = json.load(fp)
        

        if correct_antenna_level:
            antenna_level = self.get_antenna_level()

            for i in range(len(overlap_markings)):
                overlap_markings[i]['pitch'] -= antenna_level


        return overlap_markings
    


    def get_antenna_level(self):
        '''
        Load pitch points where the pseudopupils align with antenna.
        Returns Fales if no antenna level for the specimen.

        Run antenna_levels.py first to find antenna levels.
        '''
        fn = os.path.join(self.savedirs['level'], '{}.txt'.format(self.fly))
        
        if os.path.exists(fn):
            with open(fn, 'r') as fp:
                antenna_level = float(fp.read())
            
            return antenna_level
        else:
            #raise OSError('Cannot find antenna level corretion {}'.format(fn))
            return 0.

    def print_overlap(self):
        for d in self.get_overlaps():
            overlap = abs(d['horizontal_right']-d['horizontal_left'])
            line = 'Vertical {} deg: overlap width {} deg'.format(d['pitch'],
                overlap)

            print(line)
