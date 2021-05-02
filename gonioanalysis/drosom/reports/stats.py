
import numpy as np
from gonioanalysis.drosom.kinematics import magstd_over_repeats, mean_max_response


def response_magnitude(manalysers, group_name='noname',
        imagefolder_endswith=''):
    '''
    
    - group name
    - condition/stimulus name (imagefolder_endswith)
    - mean response amplitude
    - variation between animals
    - mean variation in between animals
    - the number N of animals
    '''
    
    cols = []

    respmags = []
    respstds = []

    for manalyser in manalysers:
        image_folders = manalyser.list_imagefolders(endswith=imagefolder_endswith, only_measured=True)
        
        for image_folder in image_folders:
            respmags.append( mean_max_response(manalyser, image_folder, maxmethod='mean_latterhalf') )
            respstds.append( magstd_over_repeats(manalyser, image_folder, maxmethod='mean_latterhalf') )
        

    cols.append(group_name)
    cols.append(imagefolder_endswith)
    cols.append(np.mean(respmags))
    cols.append(np.std(respmags))
    cols.append(np.mean(respstds))
    cols.append(len(manalysers))

    return cols


def response_magnitudes(grouped_manalysers, stimuli = ['uv', 'green']):
    '''
    Statistics for manalyser groups.
    See resposne_magnitude

    Arguments
    ---------
    grouped_manalysers : dict of lists of objects
        Keys are group names, items are lists that contain the manalyser
        objects.
    '''
    rows = []

    for name, manalysers in grouped_manalysers.items():
        for stimulus in stimuli:
            rows.append( response_magnitude(manalysers, group_name=name,
                    imagefolder_endswith=stimulus) )

    
    for row in rows:
        print(row)

    return rows



