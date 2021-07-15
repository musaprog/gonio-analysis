import os
import json
import warnings
import csv

import matplotlib.pyplot as plt
import numpy as np

from gonioanalysis.directories import PROCESSING_TEMPDIR, ANALYSES_SAVEDIR
from gonioanalysis.rotary_encoders import to_degrees


def load_angle_pairs(fn):
    '''
    Loading angle pairs from a file.
    
    Detached from gonio_imsoft.
    '''
    angles = []
    with open(fn, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if row:
                angles.append([int(a) for a in row])
    return angles


def load_data(folder, arl_fly=False):
    '''
    Loading a data folder.

    Returns a list where horizontal angles are grouped by the pitch angle
    and each horizontal angle is next to it's image's filename.

    grouped = [pit1, [[hor1, fn1], ...] ...]
    

    INPUT ARGUMENTS     DESCRIPTION
    arl_fly             Set true if normal DrosoX processing should be skipped
                            (meaning no outliner remove, pitch grouping etc...)

    '''
    
    if os.path.isdir(os.path.join(folder, 'rot')):
        # DrosoX format (MM + trigger): Sequence saved images + anglepairs.txt
        fns = [os.path.join(folder,'rot',fn) for fn in os.listdir(os.path.join(folder, 'rot')) if fn.endswith('.tif')]
        fns.sort()
        
        angles = load_angle_pairs(os.path.join(folder, 'anglepairs.txt'))
        
        # FIXME: Cannot really load any stack to check how many images in it,
        #       takes too long if remote filesystem
        # If saved as stack (much less images than angles)
        if 10*len(fns) < len(angles):
            fns = [fns[0]+'_{}'.format(i) for i in range(len(angles))]


    else:
        # DrosoM format (gonio imsoft): each position in own folder
        # Here for DrosoX, use only the first image in each pos folder
        # if many exists
        _folders = [f for f in os.listdir(folder) if f.startswith('pos')]
        fns = [[os.path.join(folder, f, fn) for fn in
            os.listdir(os.path.join(folder, f)) if fn.endswith('.tiff')][0] for f in _folders]
        
        angles = [ [int(n) for n in f.split('(')[1].split(')')[0].split(',')] for f in _folders]

    to_degrees(angles)
    
    print('Angles {} and images {}'.format(len(angles), len(fns)))
    if abs(len(angles) - len(fns)) > 10:
        warnings.warn("Large missmatch between the number of recorded the angles and images.", UserWarning)
    
    fns, angles = _makeSameLength(fns, angles)
   
    angles_and_images = _pitchGroupedHorizontalsAndImages(fns, angles, arl_fly=arl_fly)
    if not arl_fly:
        
        print('Determing pitches to be combined...')
        angles_to_combine = _pitchesToBeCombined(angles_and_images, angles)
        

        # GROUP NEAR PITCHES
        print('Combining pitches...')
        angles_and_images = _groupPitchesNew(angles_and_images, angles_to_combine) 
        print('After grouping: {}'.format(len(angles_and_images)))

        # NO PROBLEM AFTER THIS
        # -------------------------
        print('Removeing lonely outliners...')
        angles_and_images = _removeOutliners(angles_and_images, 2)

        #angles_and_images = self.removeShorts(angles_and_images)
        
        # SORT PITCHES
        angles_and_images.sort(key=lambda x: x[0], reverse=True)
        
       
        # SORT HORIZONTALS
        for i in range(len(angles_and_images)):
            angles_and_images[i][1].sort(key=lambda x: x[0])
        
    return angles_and_images


def _removeOutliners(angles_and_images, degrees_threshold):
    '''

    '''

    for pitch, hor_im in angles_and_images:
        remove_indices = []

        for i in range(len(hor_im)):
            center = hor_im[i][0]
            try:
                previous = hor_im[i-1][0]
            except IndexError:
                previous = None
            try:
                forward = hor_im[i+1][0]
            except IndexError:
                forward = None

            if not (previous == None and forward == None):

                if forward == None:
                    if abs(previous-center) > degrees_threshold:
                        remove_indices.append(i)
                if previous == None:
                    if abs(forward-center) > degrees_threshold:
                        remove_indices.append(i)

            #if previous != None and forward != None:
            #    if abs(previous-center) > degrees_threshold and abs(forward-center) > degrees_threshold:
            #        remove_indices.append(i)
    
        for i in sorted(remove_indices, reverse=True):
            hor_im.pop(i)

    return angles_and_images


def _getPitchIndex(pitch, angles_and_images):

    for i in range(len(angles_and_images)):
        if angles_and_images[i][0] == pitch:
            return i
    print('Warning: No pitch {} in angles_and_images'.format(pitch))
    return None


def _groupPitchesNew(angles_and_images, to_combine):
    '''
    Rotatory encoders have some uncertainty so that the pitch can "flip"
    to the next value if encoder's position in
    '''
    grouped = []

    for pitches in to_combine:
        
        combinated = []
        for pitch in pitches:
            index = _getPitchIndex(pitch, angles_and_images)

            combinated.extend(angles_and_images.pop(index)[1])
        
        grouped.append([np.mean(pitches), combinated])
        
    
    angles_and_images.extend(grouped)
    
    return angles_and_images


def _makeSameLength(lista, listb):
    if len(lista) > len(listb):
        lista = lista[0:len(listb)]
    elif len(lista) < len(listb):
        listb = listb[0:len(lista)]
    return lista, listb


def _pitchesToBeCombined(angles_and_images, angles):
    '''
    Assuming gonio scanning was done keeping pitch constant while
    varying horizontal angle, it's better to group line scans together
    because there may be slight drift in the pitch angle.
    '''
    
    pitches = [[]]
    scan_direction = -10
    
    anglefied_angles_and_images = []
    for pitch, hor_im in angles_and_images:
        for horizontal, fn in hor_im:
            anglefied_angles_and_images.append([horizontal, pitch])

    # Determine pitches that should be combined
    for i in range(1, len(angles)-1):
        if angles[i] in anglefied_angles_and_images:

            direction = np.sign( angles[i][0] - angles[i-1][0] )
            future_direction = np.sign(angles[i+1][0] - angles[i][0])
            
            if direction != scan_direction and not future_direction == scan_direction:
                pitches.append([])
                scan_direction = direction
           
            if direction == scan_direction or (scan_direction == 0 and future_direction == scan_direction):
                if not angles[i][1] in pitches[-1]:
                    pitches[-1].append(angles[i][1])

    
    pitches = [p for p in pitches if len(p)>=2 and len(p)<5]
    
   
    # A pitch can appear more than one time to be combined. This seems
    # usually happen in adjacent pitch groupings.
    # Here, combine [a,b] [b,c] -> [a,b,c]
    combine = []
    for i in range(len(pitches)-1):
        for j, pitch in enumerate(pitches[i]):
            if pitch in pitches[i+1]:                    
                combine.append([i, j])

    for i, j in sorted(combine, reverse=True):
        pitches[i].pop(j)
        pitches[i] += pitches[i+1]
        pitches.pop(i+1)
    # ----------------------------------------------------- 

    print("Pitches to be combined")
    for p in pitches:
        print(p)

    return pitches


def _pitchGroupedHorizontalsAndImages(image_fns, angles, arl_fly=False):
    '''
    Returns horizontal angles grouped by pitch (as groupHorizontals)
    but also links image fn with each horizontal angle.
    
    Note: image_fns and angles must have one to one correspondence.
    
    IDEAL STRUCTURE TO WORK WITH

    grouped = [pit1, [[hor1, fn1], ...] ...]
    '''

    grouped = []
    pitches_in_grouped = []
    
    for fn, (horizontal, pitch) in zip(image_fns, angles):
        
        if not pitch in pitches_in_grouped:
            pitches_in_grouped.append(pitch)
            grouped.append([pitch, []])
        
        i = pitches_in_grouped.index(pitch)
        grouped[i][1].append([horizontal, fn])
    
    # For each pitch angle there must be more than 10 images
    # or the whole pitch is removed
    if not arl_fly:
        grouped = [x for x in grouped if len(x[1]) > 10]
    else:
        print('ARL fly, not capping of imaging row.')

    return grouped

