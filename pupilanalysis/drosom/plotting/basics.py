'''
Most commonly needed functions to plot the data.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

from .common import vector_plot, surface_plot
from pupilanalysis.drosom.optic_flow import field_error

EYE_COLORS = {'right': 'blue', 'left': 'red'}
REPEAT_COLORS = ['green', 'orange', 'pink']


DEFAULT_ELEV = 10
DEFAULT_AZIM = 70


def plot_1d_magnitude(manalyser, image_folder=None, i_repeat=None,
        mean_repeats=False, mean_imagefolders=False, mean_eyes=False,
        color_eyes=False, gray_repeats=False, show_mean=False, show_std=False,
        show_label=True, milliseconds=False, microns=False,
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
    
    def get_x_yscaler(mag_rep_i):    
        # FIXME Pixel size and fs should be read from the data
        pixel_size = 0.816
        fs = 100
        N = len(mag_rep_i)

        if milliseconds:
            # In milliseconds
            X = 1000* np.linspace(0, N/fs, N)
        else:
            X = np.arange(N)
        
        if microns:
            yscaler = pixel_size
        else:
            yscaler = 1
        
        return X, yscaler
    
    X = None
    yscaler = None

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
            
            if X is None or yscaler is None:
                X, yscaler = get_x_yscaler(repeat_mags[0])


            for _i_repeat, mag_rep_i in enumerate(repeat_mags):
                
                N_repeats += 1
                
                if i_repeat is not None and _i_repeat != i_repeat:
                    continue
                
                
                if label:
                    if eye is None:
                        eyename = '+'.join(manalyser.eyes)
                    else:
                        eyename = eye
                    _label = label.replace('EYE', eyename).replace('ANGLE', str(angle)).replace('IREPEAT', str(_i_repeat))
                else:
                    _label = ''
                
                Y = yscaler * mag_rep_i

                if color_eyes:
                    ax.plot(X, Y, label=_label, color=EYE_COLORS.get(eye, 'green'))
                elif gray_repeats:
                    ax.plot(X, Y, label=_label, color='gray')
                else:
                    ax.plot(X, Y, label=_label)
                
                traces.append(Y)
    
    meantrace = np.mean(traces, axis=0)
    if show_mean:
        ax.plot(X, meantrace, label='mean-of-all', color='black', lw=3)

    if show_std:
        ax.plot(X, meantrace+np.std(traces, axis=0), '--', label='std-of-mean-of-all', color='black', lw=2)
        ax.plot(X, meantrace-np.std(traces, axis=0), '--', color='black', lw=2)
    
    if label and show_label:
        ax.legend(fontsize='xx-small', labelspacing=0.1, ncol=int(len(traces)/10)+1, loc='upper left')    
    

    if milliseconds:
        ax.set_xlabel('Time (ms)')
    else:
        ax.set_xlabel('Frame')

    if microns:
        ax.set_ylabel('Displacement magnitude (µm)')
    else:
        ax.set_ylabel('Displacement magnitude (pixels)')


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax, traces, N_repeats





def plot_3d_vectormap(manalyser, arrow_rotations = [0],
        guidance=False, draw_sphere=False, hide_behind=True,
        elev=DEFAULT_ELEV, azim=DEFAULT_AZIM, color=None, repeats_separately=False, vertical_hardborder=True,
        i_frame=0,
        ax=None):
    '''
    Plot an interactive 3D vectormap, where the arrows point the movement or
    feature directions.
    '''
    
    if manalyser.__class__.__name__ == 'OAnalyser' and len(arrow_rotations) == 1 and arrow_rotations[0] == 0:
        # OAnalyser specific for Drosophila; Assuming that R3-R6 line is
        # analysed, let's also draw the line from R3 to R1.
        arrow_rotations.append(29)
        i_frame = 0

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
    
    vectors = {}

    original_rotation = manalyser.vector_rotation

    if hide_behind:
        camerapos = (elev, azim)
    else:
        camerapos = False

    for i_rotation, rotation in enumerate(arrow_rotations):

        for eye in manalyser.eyes:
            if len(arrow_rotations) == 1:
                colr = EYE_COLORS[eye]
            else:
                colr = REPEAT_COLORS[i_rotation]
            
            if rotation is None or rotation == 0:
                pass
            else:
                if eye == 'left':
                    manalyser.vector_rotation = rotation
                else:
                    manalyser.vector_rotation = -rotation
            
            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True,
                    repeats_separately=repeats_separately,
                    strict=True, vertical_hardborder=vertical_hardborder)

            vector_plot(ax, *vectors_3d, color=colr, mutation_scale=10,
                    guidance=guidance,
                    draw_sphere=draw_sphere,
                    camerapos=camerapos,
                    i_pulsframe=i_frame
                    )
            
            vectors[eye] = vectors_3d
           
    manalyser.vector_rotation = original_rotation

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1, 1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(elev=elev, azim=azim)
    

    return ax, vectors


def plot_3d_differencemap(manalyser1, manalyser2, ax=None,
        elev=DEFAULT_ELEV, azim=DEFAULT_AZIM):
    '''
    Calls get_3d_vectors for both analysers. 
    
    Errors (differences) are calculated at manalyser1's points.
    '''
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
 
   
    vectors = []
    points = []

    for manalyser in [manalyser1, manalyser2]:
        lp, lvecs = manalyser.get_3d_vectors('left')
        rp, rvecs = manalyser.get_3d_vectors('right')

        points.append( np.concatenate((lp, rp)) )
        vectors.append( np.concatenate((lvecs, rvecs)) )

    errors = field_error(points[0], vectors[0], points[1], vectors[1])

    surface_plot(ax, points[0], errors)

    ax.view_init(elev=elev, azim=azim)


def compare_3d_vectormaps(manalyser1, manalyser2, axes=None,
        kwargs1={}, kwargs2={}, kwargsD={}):
    '''
    Calls get 3d vectors for both analysers
    
    manalyser1,manalyser2 : objects
        Analyser objects
    axes : list of objects
        List of length 3, holding 3 matplotlib axes
        0 is malyser 1 vectorplot, 1 for manalyser 2, and 2 is the difference.
    kwargs1,kwargs2 : dict
        List of keyword arguments to pass to `plot_3d_vectormap`
    kwargsD : dict
        List of keywords arguments to pass to `plot_3d_differencemap`
    '''


    if axes is None:
        fig = plt.figure()
        axes = []
        axes.append(fig.add_subplot(131, projection='3d'))
        axes.append(fig.add_subplot(132, projection='3d'))
        axes.append(fig.add_subplot(133, projection='3d'))
    else:
        if len(axes) < 3:
            raise ValueError('axes has to be length 3 for compare_3d_vectormaps')
    

    plot_3d_vectormap(manalyser1, ax=axes[0], **kwargs1)
    plot_3d_vectormap(manalyser2, ax=axes[1], **kwargs2)
    plot_3d_differencemap(manalyser1, manalyser2, ax=axes[2], **kwargsD)
    


