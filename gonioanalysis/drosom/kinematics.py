import os
import csv

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt



def _logistic_function(t, k, L, t0):
    '''
    The sigmoidal function.

    t : float or np.ndarray
        Independent variable, time
    k : float
        Steepness of the curve
    L : float
        Curve's maximum value
    t0: : float
        Timepoint of the mid value

    See https://en.wikipedia.org/wiki/Logistic_function
    '''
    return L / (1+np.exp(-k*(t-t0)))



def mean_max_response(manalyser, image_folder, maxmethod='max'):
    '''
    Averages over repetitions, and returns the maximum of the
    mean trace.

    manalyser
    image_folder
    maxmethod : string
        Method how the determine the maximum displacement
        'max': Take the furthest point
        'mean_latterhalf': final 50% mean discplacement
        'final' : Use the final value
    '''
    
    displacements = manalyser.get_displacements_from_folder(image_folder)
    
    mean = np.mean(displacements, axis=0)
    if maxmethod == 'max':
        return np.max(mean)
    elif maxmethod == 'mean_latterhalf':
        return np.mean(mean[int(len(mean)/2):])
    elif maxmethod == 'final':
        return mean[-1]
    else:
        raise ValueError



def magstd_over_repeats(manalyser, image_folder, maxmethod='max'):
    '''
    Standard deviation in responses
    (std of max displacement of each repeat)
    
    maxmethod : string
        See mean_max_response
    '''
    displacements = manalyser.get_displacements_from_folder(image_folder)
    
    if maxmethod == 'max':
        displacements = np.max(displacements, axis=1)
    elif maxmethod == 'mean_latterhalf':
        displacements = np.mean([d[int(len(d)/2):] for d in displacements], axis=1)

    return np.std(displacements)


def mean_topspeed(manalyser, image_folder):
    '''
    Returns the top speed of the mean response.
    '''
    mean = np.mean(manalyser.get_displacements_from_folder(image_folder), axis=0)
    
    # FIXME: Replace the loop with numpy routines
    top = 0
    for i in range(len(mean)-1):
        top = max(top, mean[i+1] - mean[i])
    
    return top


def _simple_latencies(displacements, fs, threshold=0.1):
    '''
    Interpolate and take when goes over 1/10th
    ''' 
    latencies = []

    timepoints = np.linspace(0, len(displacements[0])/fs, len(displacements[0]))
    newx = np.linspace(0, len(displacements[0])/fs, 200)

    for displacement in displacements:
        y = np.interp(newx, timepoints, displacement)
        
        index = np.where(y>(np.max(y)*threshold))[0][0]
        
        latencies.append(newx[index])
        
    return latencies



def latency(manalyser, image_folder, threshold=0.05, method='sigmoidal'):
    '''
    Response latency ie. when the response exceedes by default 5% of
    its maximum value.
    
    Arguments
    ---------
    threshold : float
        Between 0 and 1
    method : string
        Either "sigmoidal" (uses the sigmoidal fit) or
        "simple" (uses the data directly).
    
    Returns
    -------
    latency : sequency 
        The time durations in seconds that it takes for the responses
        to reach (by default) 5% of its maximum value (length of repeats).
    '''    
    fs = manalyser.get_imaging_frequency(image_folder)
    
    if method == 'simple':
        # Take the mean response of the image_folder's data
        displacements = manalyser.get_displacements_from_folder(image_folder)    
        trace = np.mean(displacements, axis=0)    
    elif method == 'sigmoidal':
        # Make a sigmoidal fit and use the sigmoidal curve
        params = sigmoidal_fit(manalyser, image_folder)
        N = len(manalyser.get_displacements_from_folder(image_folder)[0])
        time = np.linspace(0, N/fs, N)
        trace = _logistic_function(time,
                np.mean(params[1]),
                np.mean(params[0]),
                np.mean(params[2]))
    else:
        raise ValueError("method has to be 'sigmoidal' or 'simple', not {}".format(
            method))

    # Check when climbs over the threshold
    latency = _simple_latencies([trace], fs, threshold)
    return latency



def _sigmoidal_fit(displacements, fs, debug=False):
    amplitudes = []
    speeds = []
    latencies = []

    if debug:
        fig, ax = plt.subplots()

    timepoints = np.linspace(0, len(displacements[0])/fs, len(displacements[0]))

    for i_repeat, displacement in enumerate(displacements):
        
        # Initial guesses for k,L,t0
        est_L = displacement[-1]
        if est_L > 0:
            est_t0 = (np.argmax(displacement)/fs)/2
        else:
            est_t0 = (np.argmin(displacement)/fs)/2
        est_k = abs(est_L/est_t0)
        
        print('est L={} t0={} k={}'.format(est_L, est_t0, est_k))

        try:
            popt, pcov = scipy.optimize.curve_fit(_logistic_function, timepoints, displacement,
                    p0=[est_k, est_L, est_t0])
        except RuntimeError:
            # Runtime Error occurs when curve fitting takes over maxfev iterations
            # Usually then we have nonsigmoidal data (no response)
            continue
       
       
        if debug:
            ax.plot(timepoints, displacement, '-')
            ax.plot(timepoints, _logistic_function(timepoints, *popt),
                    '--', label='fit rep {}'.format(i_repeat))
            
            plt.show(block=False)
            plt.pause(.1)
            if not input('good?')[0].lower() == 'y':
                plt.close(fig)
                return None
            plt.close(fig)

        speeds.append(popt[0])
        amplitudes.append(popt[1])
        latencies.append(popt[2])
 

    return amplitudes, speeds, latencies



def sigmoidal_fit(manalyser, image_folder, figure_savefn=None):
    '''

    Assuming sigmoidal (logistic function) response.
    
    Arguments
    ---------
    manalyser : object
    image_folder : string
    figure_savefn : string
        If given, saves a figure of the fit

    Returns
    -------
    amplitudes, speeds, halfrise_times : list or None
        All Nones if image_folder has not movements
    '''

    if figure_savefn is not None:
        fig, ax = plt.subplots()


    amplitudes = []
    speeds = []
    halfrise_times = []
    
    pcovs = []
    
    displacements = manalyser.get_displacements_from_folder(image_folder)
    if not displacements:
        # Probably movements not measured
        return None, None, None

    fs = manalyser.get_imaging_frequency(image_folder)

    timepoints = np.linspace(0, len(displacements[0])/fs, len(displacements[0]))

    
    for i_repeat, displacement in enumerate(displacements):
        
        amplitude, speed, halfrise_time = _sigmoidal_fit([displacement], fs)
        
        if not amplitude or not speed or not halfrise_time:
            print(f'Fit failed for repeat={i_repeat}. Data likely not sigmoidal.')
            continue


        speeds.append(speed[0])
        amplitudes.append(amplitude[0])
        halfrise_times.append(halfrise_time[0])

        if figure_savefn is not None:
            ax.plot(timepoints, displacement, '-')
            ax.plot(timepoints, _logistic_function(timepoints, *popt),
                    '--', label='fit rep {}'.format(i_repeat))
    

    if figure_savefn:
        fig.savefig(figure_savefn)

    return amplitudes, speeds, halfrise_times



def save_sigmoidal_fit_CSV(analysers, savefn, save_fits=False, with_extra=True,
        microns=True):
    '''
    Takes in analysers, performs sigmoidal_fit for each and all image_folders.
    Then saves the results as a CSV file, and by default fit images as well.

    analysers : list of objects
        List of analyser objects
    savefn : string
        Filename.
    save_fits : bool
        Save png images of the fits.
    '''
    
    with open(savefn, 'w') as fp:
        writer = csv.writer(fp)
    
        writer.writerow(['#'])
        writer.writerow(['# Kinematics by sigmoidal fit', 'L / (1+np.exp(-k*(t-t0)))'])
        writer.writerow(['# t0 latency (s)', 'k rise speed (pixels/s)', 'L response amplitude (pixels)'])
        writer.writerow(['#'])

        if microns:
            y_units = 'pixels'
        else:
            y_units = 'µm'

        header = ['Name', 'Image folder',
                'Mean L ({})', 'STD L ({})',
                'Mean k ({}/s)', 'STD k ({}/s)',
                'Mean t0 (s)', 'STD t0 (s)', 'Fit image']

        if with_extra:
            header.extend( ['Mean max amplitude ({})',
                'Mean final amplitude ({})' ,
                'Mean latency (s)',
                'Top speed ({}/s)'] )

        writer.writerow([text.format(y_units) for text in header])
        
        i_fit = 0

        for analyser in analysers:
            
            if save_fits:
                dirname = os.path.dirname(savefn)
                folder = os.path.basename(dirname)
                figure_savefn = os.path.join(dirname, folder+'_fits', 'fit_{0:07d}.png'.format(i_fit))
                
                if i_fit == 0:
                    os.makedirs(os.path.dirname(figure_savefn), exist_ok=True)

            else:
                figure_savefn = False

            for image_folder in analyser.list_imagefolders():
                
                amplitudes, speeds, halfrise_times = sigmoidal_fit(analyser, image_folder,
                        figure_savefn=figure_savefn)
                
                if microns:
                    scaler = analyser.get_pixel_size(image_folder)
                else:
                    scaler = 1
                
                if figure_savefn:
                    figure_savefn = os.path.basename(figure_savefn)

                row = [analyser.folder, image_folder,
                        np.mean(amplitudes)*scaler, np.std(amplitudes)*scaler,
                        np.mean(speeds)*scaler, np.std(speeds)*scaler,
                        np.mean(halfrise_times), np.std(halfrise_times),
                        figure_savefn]

                if with_extra:
                    max_amplitude = scaler * mean_max_response(analyser, image_folder, maxmethod='max')
                    end_amplitude = scaler * mean_max_response(analyser, image_folder, maxmethod='final')
                    latency_value = np.mean(latency(analyser, image_folder))
                    
                    top_speed = scaler * mean_topspeed(analyser, image_folder)

                    row.extend([max_amplitude, end_amplitude, latency_value, top_speed])

                writer.writerow(row)
        
                i_fit += 1

    return None

