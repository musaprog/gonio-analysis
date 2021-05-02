'''
Functions to create plots using matplotlib out of
XAnalyser objects.
'''

import os
import json
import warnings

import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches

from numpy import sign, mean
import numpy as np
import scipy.interpolate

from gonioanalysis.directories import PROCESSING_TEMPDIR, ANALYSES_SAVEDIR


def plot_1d_overlap(xanalysers):
    '''
    Plot binocular overlap so that the x-axis is the pitch angle (vertical angle)
    and the y-axis is the width of binocular overlap.

    xanalysers      List of XAnalyser objects

    Returns the created figure.
    '''
    X = []
    Y = []
    flies = []
    
    figure = plt.figure()

    for xanalyser in xanalysers:

        X.append([])
        Y.append([])

        for marking in sorted(xanalyser.get_overlaps(), key=lambda x: x['pitch']):
            Y[-1].append(abs(marking['horizontal_left'] - marking['horizontal_right']))
            X[-1].append(marking['pitch'])
        
        flies.append(xanalyser.fly)
    

    # Interpolate the data; Required for the mean traces
    # ----------------------------------------------------
    intp_step = 1

    XXs_span = np.arange(int(np.min(np.min(X))/intp_step)*intp_step, int(np.max(np.max(X))/intp_step)*intp_step, intp_step)

    XX = []
    YY = []

    for fly, x, y in zip(flies, X,Y):
        xx = np.arange(int(np.min(x)/intp_step)*intp_step, int(np.max(x)/intp_step)*intp_step, intp_step)
        yy = np.interp(xx, x, y)
        plt.scatter(xx, yy, s=5)
        XX.append(xx)
        YY.append(yy)
    

    # Mean trace
    # ----------------
    mean_YY = []

    for x in XXs_span:
        yys_to_average = []
        
        for yy, xx in zip(YY, XX):
            try:
                index = list(xx).index(x)
            except:
                continue
            
            yys_to_average.append(yy[index])
        
        if yys_to_average:
            mean_YY.append(np.mean(yys_to_average))
        else:
            mean_YY.append(0)
    

    plt.plot(XXs_span, mean_YY, linewidth=2)
    plt.xlabel('Vertical angle (degrees)')
    plt.ylabel('Binocular overlap (degrees)')

    return figure


def plot_matrix_overlap(xanalysers):
    '''
    Plot the binocular overlap in a kind of "matrix representation" by discreting
    the continues data further, and plotting colored squared.
    '''

    def _plotMatrix(matrix, newfig=False, subplot=111):
        '''
        Temporarily hidden here. Needs commeting.
        '''
        matrix_height, matrix_width = matrix.shape
        
        if newfig == True:
            plt.figure()
        plt.subplot(subplot)

        plt.imshow(matrix, cmap='coolwarm', interpolation='none',
                extent=(hor_range[0], hor_range[1], ver_range[1], ver_range[0]),
                aspect='auto')

        ax = plt.gca();
        ax.set_xticks(np.arange(hor_range[0]+hor_step, hor_range[1]+hor_step, hor_step), minor=True)
        ax.set_yticks(np.arange(ver_range[0]+ver_step, ver_range[1]+ver_step, ver_step), minor=True) 
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)


    hor_step = 1
    ver_step = 4
    hor_range = (-40, 40)
    ver_range = (-90, 90)
    
    matrices = []

    matrix_width = int((hor_range[1]-hor_range[0])/hor_step)
    matrix_height = int((ver_range[1]-ver_range[0])/ver_step)

    for xanalyser in xanalysers:
        fly = xanalyser.fly
        markings = xanalyser.get_overlaps()

        X = []
        Y = []
        midpoints =  [] 
        
        # FIXME or not?
        # The following code for the matrix overlap plot goes quite verbose.
        # Maybe it could be made mode readable? At least commenting whats
        # going on in each section. Seems to work never the less.

        for marking in markings:
            
            mid = marking['horizontal_middle']
            marking['horizontal_left'] - mid
            
            row = []

            marking['horizontals'].sort()
            
            if marking['horizontal_right'] - marking['horizontal_left'] < hor_step:
                row = [0]*matrix_width
                row[int(matrix_width/2)] = 2

            else:
                for angle in range(hor_range[0], hor_range[1], hor_step):
                    if angle < marking['horizontal_left']:
                        row.append(0)
                    elif marking['horizontal_left'] <= angle <= marking['horizontal_right']:
                        row.append(1)
                    elif marking['horizontal_right'] < angle:
                        row.append(0)
            
            midpoints.append( (marking['horizontal_middle']) / (int(hor_range[1]-hor_range[0])/2) )

            if len(row) != matrix_width:
                print(row)
                print(marking['horizontal_left'])
                print(marking['horizontal_right'])
                raise UserWarning('Row length {} but matrix width {}'.format(len(row), matrix_width))
      

            X.append(row)
            Y.append(marking['pitch'])
        
        matrix = np.zeros( (int((ver_range[1]-ver_range[0])/ver_step), int((hor_range[1]-hor_range[0])/hor_step)) )
        matrix_i_midpoint = int(matrix.shape[0] / 2)
        for j, pitch in enumerate(range(*ver_range, ver_step)):

            indices = [y for y in Y if pitch-ver_step/2 <= y <= pitch+ver_step/2]
            indices = [Y.index(y) for y in indices]
            
            for index in indices:
                i_midpoint = int((midpoints[index])*int(matrix_width/2))
                shift = -1*(i_midpoint)
                if shift >= 0:
                    matrix[j][shift:] += np.asarray(X[index])[0:matrix_width-shift]
                elif shift < 0:
                    matrix[j][0:matrix_width+shift] += np.asarray(X[index])[-shift:]
                    
                matrix = np.round(matrix)
                matrix = np.clip(matrix, 0, 1) 
        matrices.append(matrix)
        
    avg_matrix = matrices[0] / len(matrices)
    for i in range(1, len(matrices)):
        avg_matrix += matrices[i] / len(matrices)
    
    
    matrix = np.round(avg_matrix)
    
    _plotMatrix(avg_matrix, newfig=True)

    for j in range(0, avg_matrix.shape[0]):
        row_max = np.max(avg_matrix[j])
        if row_max > np.min(avg_matrix[j]):
            avg_matrix[j] /= row_max
    
    figure = plt.figure() 
    
    #FIXME
    '''
    for i, matrix in enumerate(matrices):
        for j in range(0, matrix.shape[0]):
            if not np.any(matrix[j]):
                matrix[j] += 0.5

        _plotMatrix(matrix, subplot=int('{}{}{}'.format(3,round(len(matrices)/3),i+1)))
    '''
    _plotMatrix(avg_matrix, newfig=True)

    # Phiuw, we're out.
    return figure


def plot_experiment_illustration(xanalyser):
    '''
    Plot a video of hor the fly was rotated while simultaneously reconstructing
    the matrix plot.
    '''
    
    dpi = 300

    savepath = os.path.join(ANALYSES_SAVEDIR, 'binocular_overlap_illustration', xanalyser.fly)
    os.makedirs(savepath, exist_ok=True)

    hor_step = 1
    ver_step = 4
    hor_range = (-40, 40)
    ver_range = (-90, 90)
    
    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    ax = fig.add_subplot(121)
    #ax.set_axis_off()
    ax.set_xlim(*hor_range)
    ax.set_ylim(*ver_range)
    
    imax = fig.add_subplot(122)
    imax.set_axis_off()
    
    markings = sorted(xanalyser.get_overlaps(), key=lambda x: x['pitch'])

    # Variables to keep track of the steps    
    visited_pitches = []
    i_image = 0
    direction = True
    
    antenna_level = xanalyser.get_antenna_level()

    for pitch, horizontals_images in xanalyser.get_data():
        

        visited_horizontals = []

        if visited_pitches and abs(pitch - visited_pitches[-1]) < ver_step:
            continue
        
        print('Pitch {}'.format(pitch))

        # Find the marking of this pitch
        for marking in markings:
            if marking['pitch'] == pitch - antenna_level:
                break
        
        hm = marking['horizontal_middle']
        hl = marking['horizontal_left']
        hr = marking['horizontal_right']
    
        if visited_pitches:
            if pitch > visited_pitches[-1]:
                y = visited_pitches[-1] + ver_step/2 - 0.1
                h = (pitch + ver_step/2) - y
            else:
                print('naan')
                y2 = visited_pitches[-1] - ver_step/2
                y1 = pitch - ver_step/2
                y = y1
                h = abs(y2 - y1)
        else:
            y = pitch - ver_step/2
            h = ver_step

        if direction:
            horizontals_images.reverse()
            direction = False
        else:
            direction = True

        for horizontal, image_fn in horizontals_images:
            
            if visited_horizontals and abs(horizontal - visited_horizontals[-1]) < hor_step:
                continue
           
            if not hor_range[0] <= horizontal <= hor_range[1]:
                continue

            # Decide color of drawing
            if hl <= horizontal <= hr:
                color = "purple"
            elif horizontal < hl:
                color = "red"
            elif horizontal > hr:
                color = 'blue'
            
            if visited_horizontals:
                if horizontal > visited_horizontals[-1]:
                    x = visited_horizontals[-1] + hor_step/2
                    w = (horizontal + hor_step/2) - x
                else:
                    x2 = visited_horizontals[-1] - hor_step/2
                    x1 = horizontal - hor_step/2
                    x = x1
                    w = x2 - x1
            else:
                x = horizontal - hor_step/2
                w = hor_step
            
            
            rect = matplotlib.patches.Rectangle((-(x+w-hm), -(y+h-antenna_level)), w, h, color=color)
            ax.add_patch(rect)

            image = tifffile.imread(image_fn)
            image = np.clip(image, 0, np.percentile(image, 95))
            image = np.rot90(image) / np.max(image)

            try:
                imshow_obj.set_data(image)
            except UnboundLocalError:
                imshow_obj = imax.imshow(image, cmap='gray')
            
                
            fig.savefig(os.path.join(savepath, 'image_{:08d}'.format(i_image)), dpi=dpi)
            i_image += 1
            
            
            
            visited_horizontals.append(horizontal)
        
        visited_pitches.append(pitch)



    

# FIXME or remove?
'''
def plot2DOverlap(xanalysers):
    
    Xl = []
    Yl = []
    Xr = []
    Yr = []
    Xm = []
    Ym = []



    for marking in self.overlap_markings:
        
        mid = marking['horizontal_middle']
        #mid = 0
        Xl.append(marking['horizontal_left']-mid)
        Xr.append(marking['horizontal_right']-mid)
        Xm.append(marking['horizontal_middle']-mid)
        Yl.append(marking['pitch'])
        Yr.append(marking['pitch'])
        Ym.append(marking['pitch'])
    
    plt.scatter(Xl, Yl, color='blue')
    plt.scatter(Xr, Yr, color='red')
    plt.scatter(Xm, Ym, color='yellow')


    plt.show()

   
def plotFancy2DOverlap():

    X,Y,C = [[],[],[]]
    for marking in self.overlap_markings:

        mid = marking['horizontal_middle']
        
        for i in range(len(marking['horizontals'])):
            
            pitch = marking['pitch']
            horizontal = marking['horizontals'][i]
            L = min(marking['horizontal_left'], marking['horizontal_right'])
            R = max(marking['horizontal_left'], marking['horizontal_right'])
            
            if L < horizontal < R:
                C.append(2)
            else:
                C.append(1)
            
            X.append(horizontal-mid)
            Y.append(pitch)

    f_int = scipy.interpolate.interp2d(X, Y, C, fill_value=1)
    
    X = np.linspace(np.min(X), np.max(X), 100)
    Y = np.linspace(np.min(Y), np.max(Y), 100)
    C = f_int(X, Y)
    
    X, Y = np.meshgrid(X, Y)
    
    C = np.around(C)

    plt.pcolormesh(X, Y, C)

    plt.show()
'''

