'''
Finding where the pseudopupils align with the antenna (binary_search way).
This pitch value can be later taken as a zero pitch.

TODO  - architectual choice: merge drosox and drosom or separate
        all drosom/drosox code to respective folders

'''

import os

import numpy as np
import matplotlib.pyplot as plt

from directories import ANALYSES_SAVEDIR
from droso import DrosoSelect
from drosox import XLoader
from drosom import MLoader
from imageshower import ImageShower
from binary_search import binarySearchMiddle
from pupil_imsoft.anglepairs import strToDegrees
from drosoalr import loadReferenceFly

from imalyser.matching import MatchFinder



class AntennaLevelFinder:
    '''
    Ask user to find the antenna levels
    '''
    
    
    def analyseFly(self, folder):
        '''
        Call this to make user to select antenna levels.
        '''
        
        fly = os.path.split(folder)[1]
        
        if os.path.exists(os.path.join(ANALYSES_SAVEDIR, 'antenna_levels', fly+'.txt')):
            print('Fly {} is already analysed. Redo (y/n)?'.format(fly))
            if not input('>> ').lower() in ['yes', 'y']:
                return False

        if 'DrosoX' in fly or 'DrosoALR' in fly:
            # If DrosoX, use drosox loader and find antenna levels by user
            # driven binary search. 
            
            fig, ax = plt.subplots()
            shower = ImageShower(fig, ax)
            
            if 'DrosoALR' in fly:
                arl = True
            else:
                arl = False

            pitches, images = self._drosox_load(folder, arl)
            
            shower.setImages(images)
            center = binarySearchMiddle(len(images), shower)
            
            shower.close()
            
            result = str(pitches[center])


        elif 'DrosoM' in fly:
        
            mode = ['manual', 'auto'][1]

            # DrosoM is harder, there's images every 10 degrees in pitch.
            # Solution: Find closest matches using analysed DrosoX data
            
            # Load DrosoM data
            pitches, images = self._drosom_load(folder)
            
            
            # Load reference fly data
            reference_pitches = {fn: pitch for pitch, fn in loadReferenceFly('/work1/pupil/tmp/test').items()}
            print(reference_pitches)
            reference_images = list(reference_pitches.keys())
            reference_images.sort()

            fig1, ax1 = plt.subplots()
            ref_shower = ImageShower(fig1, ax1)
            ref_shower.setImages(reference_images)
            
            fig2, ax2 = plt.subplots()
            m_shower = ImageShower(fig2, ax2)
            
            #matcher = MatchFinder()

            offsets = []

            for pitch, image in zip(pitches, images):
                
                m_shower.setImages([image])
                m_shower.setImage(0)

                #best_drosox_image = matcher.findBest(image, drosox_images)
                best_drosoref_image = reference_images[ binarySearchMiddle(len(reference_images), ref_shower) ]
                 
                reference_pitch = float(reference_pitches[best_drosoref_image])
                
                print('Pitch {}, reference {}'.format(pitch, reference_pitch))

                offsets.append( pitch - reference_pitch)
            
            ref_shower.close()
            m_shower.close()

            result = np.mean(offsets)
            
            print('Reporting mean offset of {} (from {})'.format(result, offsets))


        with open(os.path.join(ANALYSES_SAVEDIR, 'antenna_levels', fly+'.txt'), 'w') as fp:
            fp.write(str(float(result)))
       
   
    
    def _load_drosox_reference(self, folder, fly):
        '''
        Load DrosoX reference for DrosoM antenna level finding.
        
        returns dictionary dict = {image_fn1: correted_pitch_1, ....}
        '''
        pitches, images = self._drosox_load(folder, True)
        
        with open(os.path.join(ANALYSES_SAVEDIR, 'antenna_levels', fly+'.txt'), 'r') as fp:
            offset = float(fp.read())

        return {image: pitch-offset for image, pitch in zip(images, pitches)}




    def _drosox_load(self, folder, arl):
        '''
        Private method, not intented to be called from outside.
        Loads pitches and images.
        '''

        xloader = XLoader()
        data = xloader.getData(folder, arl_fly=arl)

        pitches = []
        images = []


 
        # Try to open if any previously analysed data
        analysed_data = []
        try: 
            with open(os.path.join(ANALYSES_SAVEDIR, 'binary_search', 'results_{}.json'.format(fly)), 'r') as fp:
                analysed_data = json.load(fp)
        except:
            pass
        analysed_pitches = [item['pitch'] for item in analysed_data]
        

        for i, (pitch, hor_im) in enumerate(data):
            
            if pitch in analysed_pitches:
                j = [k for k in range(len(analysed_pitches)) if analysed_pitches[k]['pitch'] == pitch]
                center_index = analysed_data[j]['index_middle']
            else:
                center_index = int(len(hor_im)/2)
            
            pitches.append(pitch)
            images.append(hor_im[center_index][1])
            
        return pitches, images


    def _drosom_load(self, folder):
        '''
        Private method, not intented to be called from outside.
        Loads pitches and images.
        '''

        mloader = MLoader()
        data = mloader.getData(folder)

        pitches =[]
        images = []

        for str_angle_pair in data.keys():
            angle_pair = strToDegrees(str_angle_pair)
            if -3 < angle_pair[0] < 3:
                pitches.append(angle_pair[1])
                images.append(data[str_angle_pair][0][0])
        
        pitches, images = zip(*sorted(zip(pitches, images)))

        return pitches, images

    
def main():
    finder = AntennaLevelFinder()
    
    selector = DrosoSelect()
    folders = selector.askUser()
       
    for folder in folders:
        finder.analyseFly(folder)

if __name__ == "__main__":
    # FIXME TODO
    main()
