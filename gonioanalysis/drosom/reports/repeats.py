
import os

import numpy as np

from gonioanalysis.directories import ANALYSES_SAVEDIR
from gonioanalysis.drosom.kinematics import (
        magstd_over_repeats,
        sigmoidal_fit,
        )
from .left_right import write_CSV_cols


SAVEDIR = os.path.join(ANALYSES_SAVEDIR, 'repeats_exports')


def mean_repeats(manalysers, group_name, wanted_imagefolders=None,
        savedir=SAVEDIR):
    '''
    Here we mean the repeat 1 of all flies together, then
    the repeat 2 of all flies together, and so on.

    This is for example for the intensity series where the stimulus intensity
    increases by every repeat and we want to know the mean response
    of the flies to the flash (repeat) 1, the flash (repeat) 2, and so on.
    '''

    all_traces = []

    for manalyser in manalysers:

        if wanted_imagefolders:
            _image_folders = wanted_imagefolders.get(manalyser.name, [])
        else:
            _image_folders = manalyser.list_imagefolders()
        
        for image_folder in _image_folders:
            
            for eye in manalyser.eyes:
                traces = manalyser.get_magnitude_traces(eye,
                        image_folder=image_folder)

                traces = list(traces.values())
                if traces:
                    all_traces.append(traces[0])

    # Average repeat 1 together, repeat 2 together etc.
    mean_traces = []

    for i_repeat in range(len(all_traces[0])):
        mean = np.mean([data[i_repeat] for data in all_traces], axis=0)
        mean_traces.append(mean.tolist())
    
    os.makedirs(savedir, exist_ok=True)
    write_CSV_cols(os.path.join(savedir, group_name+'.csv'), mean_traces)



def repeat_stds(manalysers, group_name, wanted_imagefolders=None,
        savedir=SAVEDIR):
    '''
    Variation within an specimen(s) (not between specimens)
    '''
    stds = [['name', 'disp-std', 'speed-std', '1/2-time std']]
    for manalyser in manalysers:
        if wanted_imagefolders:
            _image_folders = wanted_imagefolders.get(manalyser.name, [])
        else:
            _image_folders = manalyser.list_imagefolders()
        
        for image_folder in _image_folders:
            
            std = [np.std(z) for z in sigmoidal_fit(manalyser, image_folder)]
            
            std[0] = magstd_over_repeats(manalyser, image_folder, maxmethod='mean_latterhalf')
            
            std.insert(0, manalyser.name+'_'+image_folder)

            stds.append(std)


    os.makedirs(savedir, exist_ok=True)
    write_CSV_cols(os.path.join(savedir, group_name+'.csv'), stds)

