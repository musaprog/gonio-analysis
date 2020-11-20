'''
Most commonly needed functions to plot the data.
'''

import numpy as np
import matplotlib.pyplot as plt

EYE_COLORS = {'right': 'blue', 'left': 'red'}

def plot_1d_magnitude(manalyser, image_folder=None, i_repeat=None,
        mean_repeats=False, mean_imagefolders=False, mean_eyes=False,
        color_eyes=False, show_mean=False,
        label="EYE-ANGLE-IREPEAT", ax=None):
    '''
    Plots 1D displacement magnitude over time, separately for each eye.
    
    Arguments
    ---------
    manalyser : object
        MAnalyser object instance
    image_folder : string or None
        Image folder to plot the data from. If None (default), plot all image folders.
    mean_repeats : bool
        Wheter to take mean of the repeats or plot each repeat separately
    mean_imagefolders : bool
        If True and image_folder is None (plotting all image folders), takes the mean
        of all image folders.
    mean_eyes : bool
        Wheter to take a mean over the left and right eyes
    label : string
        Label to show. If None, no label. Otherwise
        EYE gets replaced with eye
        ANGLE gets replaced by image folder name
        IREPEAT gets reaplaced by the number of repeat
    
    Returns
        ax
            Matplotlib axes
        traces
            What has been plotted
        N_repeats
            The total number of repeats (independent of i_repeat)
    '''
    

    if ax is None:
        fig, ax = plt.subplots()

    if mean_eyes:
        eyes = [None]
    else:
        eyes = manalyser.eyes

    N_repeats = 0
    traces = []

    for eye in eyes:
        magtraces = manalyser.get_magnitude_traces(eye, image_folder=image_folder,
                mean_repeats=mean_repeats, mean_imagefolders=mean_imagefolders)
        
        for angle, repeat_mags in magtraces.items():

            for _i_repeat, mag_rep_i in enumerate(repeat_mags):
                
                if i_repeat is not None and _i_repeat != i_repeat:
                    continue
                
                N_repeats += 1
                
                if label:
                    if eye is None:
                        eyename = '+'.join(manalyser.eyes)
                    else:
                        eyename = eye
                    _label = label.replace('EYE', eyename).replace('ANGLE', str(angle)).replace('IREPEAT', str(_i_repeat))
                else:
                    _label = ''
                
                if color_eyes:
                    ax.plot(mag_rep_i, label=_label, color=EYE_COLORS.get(eye, 'green'))
                else:
                    ax.plot(mag_rep_i, label=_label)
                
                traces.append(mag_rep_i)

    if show_mean:
        ax.plot(np.mean(traces, axis=0), label='mean-of-all', color='black', lw=3)

    if label:
        ax.legend(fontsize='xx-small', labelspacing=0.1, ncol=int(len(traces)/10)+1, loc='upper left')    
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Displacement sqrt(x^2+y^2) (pixels)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax, traces, N_repeats


