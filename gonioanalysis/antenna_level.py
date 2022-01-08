'''
Code paths related to manually alinging the vertical angles for many
specimens (aka. antenna level or zero correction)
'''

import os
import ast

import numpy as np
import matplotlib.pyplot as plt

from gonioanalysis.directories import ANALYSES_SAVEDIR, CODE_ROOTDIR, PROCESSING_TEMPDIR
from gonioanalysis.droso import DrosoSelect
from gonioanalysis.drosom.loading import load_data
from gonioanalysis.image_tools import ImageShower
from gonioanalysis.binary_search import binary_search_middle
from gonioanalysis.rotary_encoders import to_degrees

ZERO_CORRECTIONS_SAVEDIR = os.path.join(PROCESSING_TEMPDIR, 'vectical_corrections')


def load_reference_fly(reference_name):
    '''
    Returns the reference fly data, dictionary with pitch angles as keys and
    image filenames as items.
    '''
    pitches = []
    with open(os.path.join(ZERO_CORRECTIONS_SAVEDIR, reference_name, 'pitch_angles.txt'), 'r') as fp:
        for line in fp:
            pitches.append(line)

    images = [os.path.join(ZERO_CORRECTIONS_SAVEDIR, reference_name, fn) for fn in os.listdir(
        os.path.join(ZERO_CORRECTIONS_SAVEDIR, reference_name)) if fn.endswith('.tif') or fn.endswith('.tiff')]

    images.sort() 
    return {pitch: fn for pitch,fn in zip(pitches, images)}



#OLD def _drosom_load(self, folder):
def load_drosom(folder):
    '''
    Loads frontal images of the specified drosom
    
    folder          Path to drosom folder
    '''

    data = load_data(folder)

    pitches =[]
    images = []

    for str_angle_pair in data.keys():
        #angle_pair = strToDegrees(str_angle_pair)
        i_start = str_angle_pair.index('(')
        i_end = str_angle_pair.index(')')+1

        #print(str_angle_pair[i_start:i_end])
        angle_pair = [list(ast.literal_eval(str_angle_pair[i_start:i_end]))]
        to_degrees(angle_pair)
        angle_pair = angle_pair[0]

        if -10 < angle_pair[0] < 10:
            pitches.append(angle_pair[1])
            images.append(data[str_angle_pair][0][0])
    
    pitches, images = zip(*sorted(zip(pitches, images)))

    return pitches, images


def save_antenna_level_correction(fly_name, result):
    '''
    Saves the antenna level correction that should be a float
    '''
    directory = ZERO_CORRECTIONS_SAVEDIR
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, fly_name+'.txt'), 'w') as fp:
        fp.write(str(float(result)))
 

class AntennaLevelFinder:
    '''
    Ask user to find the antenna levels
    '''
    
    
    def find_level(self, folder):
        '''
        Call this to make user to select antenna levels.
        folder      Full path to the folder
        '''
        
        fly = os.path.split(folder)[1]
        
        #if os.path.exists(os.path.join(ANALYSES_SAVEDIR, 'antenna_levels', fly+'.txt')):
        #    print('Fly {} is already analysed. Redo (y/n)?'.format(fly))
        #    if not input('>> ').lower() in ['yes', 'y']:
        #        return False

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
            center = binary_search_middle(len(images), shower)
            
            shower.close()
            
            result = str(pitches[center])


        else:
        
            # DrosoM is harder, there's images every 10 degrees in pitch.
            # Solution: Find closest matches using analysed DrosoX data
            
            # Load DrosoM data
            pitches, images = self._drosom_load(folder)
            
            
            # Load reference fly data
            reference_pitches = {fn: pitch for pitch, fn in load_reference_fly('alr_data').items()}
            #print(reference_pitches)
            reference_images = list(reference_pitches.keys())
            reference_images.sort()

            fig1, ax1 = plt.subplots()
            fig1.canvas.set_window_title('Reference Drosophila')
            ref_shower = ImageShower(fig1, ax1)
            ref_shower.setImages(reference_images)
            
            fig2, ax2 = plt.subplots()
            fig2.canvas.set_window_title('{}'.format(fly))
            m_shower = ImageShower(fig2, ax2)
            
            offsets = []

            for pitch, image in zip(pitches, images):
                
                m_shower.setImages([image])
                m_shower.setImage(0)

                #best_drosox_image = matcher.findBest(image, drosox_images)
                best_drosoref_image = reference_images[ binary_search_middle(len(reference_images), ref_shower) ]
                 
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
        
        with open(os.path.join(ZERO_CORRECTIONS_SAVEDIR, fly+'.txt'), 'r') as fp:
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




def main():
    finder = AntennaLevelFinder()
    
    selector = DrosoSelect()
    folders = selector.askUser()
       
    for folder in folders:
        finder.find_level(folder)

if __name__ == "__main__":
    # FIXME TODO
    main()
