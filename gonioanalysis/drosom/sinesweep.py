'''
Analysing sinusoidal sweep data.

Currently supports Gonio Imsoft flash_type's 
'''

import os
import csv
import math

import numpy as np
import scipy.stats
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt


from gonioanalysis.directories import ANALYSES_SAVEDIR

def _get_stimulus(flash_type, t, fs):
    '''
    
    flash_type : strings
        Flash_type used in Gonio Imsoft
    t : float
        Duration of the stimulus
    fs : float
        Sampling rate of the stimulus
    '''

    if ',' in flash_type:
        flash_type, f0, f1 = flash_type.split(',')
        f0 = float(f0)
        f1 = float(f1)
    else:
        f0 = 0.5
        f1 = 100

    
    timepoints = np.linspace(1/fs, t, int(t*fs))

    stimulus_frequency = f0 * (f1/f0) ** (timepoints/t)
    stimulus_amplitude = scipy.signal.chirp(timepoints, f0=f0, t1=t, f1=f1,
            phi=-90, method='logarithmic')
    if flash_type == 'squarelogsweep':
        stimulus_amplitude[stimulus_amplitude>=0] = 1
        stimulus_amplitude[stimulus_amplitude<0] = -1
    elif flash_type == '3steplogsweep':
        cstep = np.sin(np.pi/4)
        stimulus_amplitude[np.abs(stimulus_amplitude) <= cstep] = 0
        stimulus_amplitude[stimulus_amplitude > cstep] = 1
        stimulus_amplitude[stimulus_amplitude < -cstep] = -1
    else:
        pass

    stimulus_amplitude = (stimulus_amplitude+1)/2

    return timepoints, stimulus_frequency, stimulus_amplitude


from numpy.fft import fftshift

def _find_zeroindices(stimulus):
    '''
    Poor mans iterative zero point finding.
    '''
    zero_indices = []

    previous_dir = 0

    for i in range(len(stimulus)):
        dirr = np.sign(stimulus[i] - stimulus[i-1])
        
        if dirr != previous_dir:
            zero_indices.append(i)

            previous_dir = dirr
        else:
            pass
    return zero_indices


def _sham_frequency_response(stimulus_timepoints, stimulus_frequencies, stimulus, response,
        interpolate=True):
    '''
    Calculates frequency response to sinusoidal sweep signal.
    
    interpolate : bool
        Interpolate to stimulus_frequencies
    '''
    #fn = '/home/joni/.gonioanalysis/final_results/sineweep_analysis/wtb_sinusoidal_07_right_sinelogsweep.csv'
    #fn = '/home/joni/.gonioanalysis/final_results/sineweep_analysis/wtb_sinusoidal_07_right_squarewave_cam200Hz.csv'
    #data = np.loadtxt(fn,
    #        skiprows=2, delimiter=',').T

    #print(data)

    #stimulus_timepoints = data[1][:-5]
    #stimulus_frequencies = data[2][:-5]
    #stimulus = data[3][:-5]
    #response = data[4][:-5]

    fs = 1 / (stimulus_timepoints[1]-stimulus_timepoints[0])

    #b, a = scipy.signal.butter(1, 0.5, 'high', fs=fs)
    #response = scipy.signal.filtfilt(b, a, response)

    #calculate_frequency_response(stimulus_timepoints, stimulus)


    cut_indices = _find_zeroindices(stimulus)[1::2]

    #for indx in cut_indices:
    #    plt.plot(2*[stimulus_timepoints[indx]], [0, 1], '--', color='black')

    #plt.plot(stimulus_timepoints, response, color='red')
    #plt.plot(stimulus_timepoints, stimulus, color='blue')
    #plt.show()

    freqs = []
    resps = []

    for i1, i2 in zip(cut_indices[0:-1], cut_indices[1:]):
        #plt.plot(stimulus_timepoints[i1:i2], stimulus[i1:i2], color='blue')
        #plt.plot(stimulus_timepoints[i1:i2], response[i1:i2], color='red')
        #plt.plot(stimulus_timepoints, response2, color='orange')
        #plt.show()
        chunk = response[i1:i2]
        resp = max(chunk) - min(chunk)
        
        freq = np.mean(stimulus_frequencies[i1:i2])

        resps.append(resp)
        freqs.append(freq)
    
    #plt.plot(freqs, resps)
    #plt.xscale('log')
    #plt.show()

    if interpolate:
        f = scipy.interpolate.interp1d(freqs, resps, fill_value='extrapolate', bounds_error=False)
        resps = f(stimulus_frequencies)
        freqs = stimulus_frequencies
    
    return freqs, resps



def save_sinesweep_analysis_CSV(analysers, debug=False):
    '''
    Save X and Y components of the movement in a csv file
    for the selected manalysers.

    Columns
        i_camframe time stimulus_fs mean_response response_rep1, response_rep2...
    '''

    stimuli = {}

    savedir = os.path.join(ANALYSES_SAVEDIR, 'sineweep_analysis')
    os.makedirs(savedir, exist_ok=True)
   
    final_cols = {}

    if debug:
        fig, ax = plt.subplots()

    for analyser in analysers:
        for eye in analyser.eyes:

            for image_folder in analyser.list_imagefolders():
                
                imagefolder_key = image_folder[3:]

                try:
                    N_repeats = len(analyser.movements[eye][imagefolder_key])
                except KeyError:
                    print('No data for {}'.format(analyser))
                    continue

                # Get imaging parameters
                im_fs = analyser.get_imaging_frequency(image_folder)

                im_params = analyser.get_imaging_parameters(image_folder)
                flash_type = im_params.get('flash_type', '')
               
                if flash_type.split(',')[0] not in ['squarelogsweep', 'sinelogsweep', '3steplogsweep']:
                    print('Unkown flash_type {}, skipping'.format(flash_type))
                    continue


                # Get movement data
                pixel_size = analyser.get_pixel_size(image_folder)
                print('N_repeats {}'.format(N_repeats))
                Xs = [np.array(analyser.movements[eye][imagefolder_key][i_repeat]['x'])*pixel_size for i_repeat in range(N_repeats)]
                Ys = [np.array(analyser.movements[eye][imagefolder_key][i_repeat]['y'])*pixel_size for i_repeat in range(N_repeats)]

                #mean_displacement = np.mean([np.sqrt(np.array(X)**2+np.array(Y)**2) for X, Y in zip(Xs, Ys)], axis=0)

                meanX = np.mean(Xs, axis=0)
                meanY = np.mean(Ys, axis=0)
                meanX -= meanX[0]
                meanY -= meanY[0]

                mean_displacement = np.sqrt(meanX**2+meanY**2)
                
                #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(meanX, meanY)
                
                # Rotate fit along xaxis
                #rot = -math.atan(slope)
                #c = math.cos(rot)
                #s = math.sin(rot)
                #pc1 = [c*x - s*y for x,y in zip(meanX, meanY)]
                #pc2 = [s*x + c*y for x,y in zip(meanX, meanY)]
                
                # Stimulus frequencies
                #stim_fs = 1000
                

                timepoints, stimulus_frequency, stimulus_amplitude = _get_stimulus(flash_type, len(Xs[0])/im_fs, im_fs)
                
                if flash_type not in stimuli:
                    # FIXME 1kHz output in Imsoft but information not saved
                    stim_fs = 1000
                    dense = _get_stimulus(flash_type, len(Xs[0])/im_fs, stim_fs)
                    
                    stimuli[flash_type] = dense

                # "Frequency response"
                fr_freqs, fr = _sham_frequency_response(timepoints, stimulus_frequency,
                        stimulus_amplitude, mean_displacement, interpolate=True)

                # Save csv

                fn = '{}_{}_{}.csv'.format(analyser.folder, eye, im_params['suffix'])
                
                if debug:
                    ax.clear()
                    ax.scatter(meanX, meanY)
                    ax.plot(meanX, intercept + slope*meanX)
                    fig.savefig(os.path.join(savedir, fn.replace('.csv', '.png')))

                if final_cols.get(flash_type, None) is None:
                    final_cols[flash_type] = {'time (s)': [], 'f_stimulus (Hz)': [],
                            'displacement': [], 'frequency_response': [], 'specimen_name': []}
                

                final_cols[flash_type]['time (s)'].append(timepoints)
                final_cols[flash_type]['f_stimulus (Hz)'].append(stimulus_frequency)
                final_cols[flash_type]['displacement'].append(mean_displacement)
                final_cols[flash_type]['specimen_name'].append(analyser.folder)
                final_cols[flash_type]['frequency_response'].append(fr)
                

                with open(os.path.join(savedir, fn), 'w') as fp:
                    writer = csv.writer(fp)
                    
                    writer.writerow(['Displacement = sqrt(X_mean**2+Y_mean**2)'])
                    
                    row = ['i_camframe',
                            'time (s)',
                            'f_stimulus (Hz)',
                            'instantaneous stimulus amplitude (0-1)',
                            #'pc1 (µm)',
                            #'displacement',
                            'displacement2 (µm)',
                            #'pc2 (µm)'
                            #'X_mean (µm)',
                            #'Y_mean (µm)',
                            #'Displacement (µm)'
                            'Frequency response (µm)']

                    for i_repeat, (x, y) in enumerate(zip(Xs, Ys)):
                        row.append('X rep_{}'.format(i_repeat))
                        row.append('Y rep_{}'.format(i_repeat))

                    writer.writerow(row)

                    for i in range(len(Xs[0])-10):
                        
                        row = []
                        row.append(i)
                        row.append(timepoints[i])
                        row.append(stimulus_frequency[i])
                        row.append(stimulus_amplitude[i])
                        #row.append(pc1[i])
                        #row.append(pc2[i])
                        #row.append(meanX[i])
                        #row.append(meanY[i])
                        #row.append(math.sqrt(meanX[i]**2+meanY[i]**2))
                        row.append(mean_displacement[i])
                        row.append(fr[i])

                        for x, y in zip(Xs, Ys):
                            row.append(x[i])
                            row.append(y[i])

                        writer.writerow(row)
                    
    
    for flash_type in final_cols:
        with open(os.path.join(savedir, 'mean_'+flash_type+'.csv'), 'w') as fp:
            writer = csv.writer(fp)
            
            row = []
            row.append('i_camframe')
            row.append('time (s)')
            row.append('f_stimulus (Hz)')
            
            N = len(final_cols[flash_type]['displacement'])
            
            for k in range(N):
                row.append('{} response (µm)'.format(final_cols[flash_type]['specimen_name'][k]))
            
            row.append('mean response (µm)')

            for k in range(N):
                row.append('frequency response (µm)')


            row.append('mean frequency response (µm)')

            writer.writerow(row)

            for i in range(len(final_cols[flash_type]['time (s)'][0])-10):

                row = []
                row.append(i)
                row.append(final_cols[flash_type]['time (s)'][0][i])
                row.append(final_cols[flash_type]['f_stimulus (Hz)'][0][i])
                

                displacements = []
                for j in range(N):
                    row.append(final_cols[flash_type]['displacement'][j][i])
                    displacements.append(row[-1])

                row.append(np.mean(displacements))
                

                fr = []
                for j in range(N):
                    row.append(final_cols[flash_type]['frequency_response'][j][i])
                    fr.append(row[-1])
                row.append(np.mean(fr))

                writer.writerow(row)
    
    for stimulus in stimuli:

        with open(os.path.join(savedir, 'stimulus_{}.csv'.format(stimulus)), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['Time (ms)', 'Frequency (Hz)', 'Amplitude (V)'])
            for i in range(len(stimuli[stimulus][0])):
                writer.writerow([data[i] for data in stimuli[stimulus]])
                

