'''
Extension of find antenna levels to create a reference fly.

This script has to be (ideally) run only once on DrosoALR flies, creating
an antenna lvl reference fly that can be used for other flies as well.
'''

import os

import numpy as np
import tifffile

from gonioanalysis.antenna_level import AntennaLevelFinder
from gonioanalysis.directories import PROCESSING_TEMPDIR_BIGFILES

from imalyser.averaging import Templater
from imalyser.aligning import Aligner
from imalyser.common import imwrite


DROSO_DATADIR = input('Input data directory >> ')


def loadReferenceFly(folder):
    '''
    Returns the reference fly data, dictionary with pitch angles as keys and
    image filenames as items.
    '''
    pitches = []
    with open(os.path.join(folder, 'pitch_angles.txt'), 'r') as fp:
        for line in fp:
            pitches.append(line)

    images = [os.path.join(folder, fn) for fno in s.listdir(folder) if fn.endswith('.tif') or fn.endswith('.tiff')]
    
    return {pitch: fn for pitch,fn in zip(pitches, fns)}


class ReferenceCreator:
    '''
    Create a refenrence fly that can be loaded like an ALR (antenna level reference) fly.
    '''
    
    def __init__(self, name):
        self.name = name
        self.savedir = 'alr_data'
        os.makedirs(self.savedir, exist_ok=True)

    def _loadData(self):
        '''
        Loads all the present ALR flies.
        '''
        
        reference_data = []

        
        alr_flies = [fn for fn in os.listdir(DROSO_DATADIR) if 'DrosoALR' in fn] 
        alr_folders = [os.path.join(DROSO_DATADIR, fn)for fn in alr_flies] 

        lfinder = AntennaLevelFinder()
        
        for folder,fly in zip(alr_folders, alr_flies):    
            reference_data.append( lfinder._load_drosox_reference(folder, fly) )
        
        return reference_data



    def _restructureData(self, reference_data):
        '''
        Restructures data from self._loadData to work in self.createReferenceFly.
        Limits that are not present in all the flies are removed (FIXME structure this phrase better).
        
        Input data =  [fly1_dict, fly2_dict, ...] where
                flyi_dict = {image_fn_1: pitch_1, ...}

        Output data = {ptich_1: [image_fn_1_fly1,...]}
        '''
        restructured_data = {}

        step_size = 1 # degree
        N = len(reference_data)
        
        # 1) MAKE ALL FLIES TO SPAN THE SAME ANGLES
        angles = [[angle for image_fn, angle in reference_data[i].items()] for i in range(N)]

        mins = [np.min(angles[i]) for i in range(N)]
        maxs = [np.max(angles[i]) for i in range(N)]

        limits = (int(np.max(mins)), int(np.min(maxs)))



        # 2)  BIN together based on step_size
        
        for iter_pitch in range(*limits, step_size):
            
            restructured_data[str(iter_pitch)] = []

            for fly_i_data in reference_data:
                for image_fn, pitch in fly_i_data.items():
                    
                    if iter_pitch <= pitch < iter_pitch + step_size:
                        restructured_data[str(iter_pitch)].append( image_fn )
            
        return restructured_data

    def createReferenceFly(self):
        
        data = self._loadData()
        data = self._restructureData(data)

        templater = Templater()
        aligner = Aligner()

        templates = []
        angles_order = []

        for i, (angle, images) in enumerate(sorted(data.items(), key=lambda x:float(x[0]))):
            
            print('Templating for pitch {} degrees ({}/{}), {} images to align and average together.'.format(angle, i+1, len(data), len(images)))

            template = templater.template(images)
            templates.append(template)
            #imwrite(fn, template)

            angles_order.append(angle)

        print('Aligning all template images...')
        offsets = aligner.calcOffsets(templates)
        templates = aligner.getAligned(templates, offsets)
        
        for i, template in enumerate(templates):
            

            fn = os.path.join(self.savedir, 'im{:0>5}.tif'.format(i))
            print('Saving {}'.format(fn))
            tifffile.imwrite(fn, template)

        with open(os.path.join(self.savedir, 'pitch_angles.txt'), 'w') as fp:
            for pitch in angles_order:
                fp.write(pitch+'\n')




def main():
    
    creator = ReferenceCreator('Drosophila')
    creator.createReferenceFly()

if __name__ == "__main__":
    main()
