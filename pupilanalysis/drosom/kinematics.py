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
    '''
    
    displacements = manalyser.get_displacements_from_folder(image_folder)
    
    mean = np.mean(displacements, axis=0)
    if maxmethod == 'max':
        return np.max(mean)
    elif maxmethod == 'mean_latterhalf':
        return np.mean(mean[int(len(mean)/2):])
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



def sigmoidal_fit(manalyser, image_folder, figure_savefn=None, debug=False):
    '''

    Assuming sigmoidal (logistic function) response.
    
    Returns the following lists
        amplitudes, speeds, latencies
    '''

    if figure_savefn is not None or debug:
        fig, ax = plt.subplots()


    amplitudes = []
    speeds = []
    latencies = []
    
    pcovs = []
    
    displacements = manalyser.get_displacements_from_folder(image_folder)
    fs = manalyser.get_imaging_frequency(image_folder)

    timepoints = np.linspace(0, len(displacements[0])/fs, len(displacements[0]))

    for i_repeat, displacement in enumerate(displacements):
        
        # Initial guesses for k,L,t0
        est_t0 = (np.argmax(displacement)/fs)/2
        est_L = np.max(displacement)
        est_k = est_L/est_t0
        
        try:
            popt, pcov = scipy.optimize.curve_fit(_logistic_function, timepoints, displacement,
                    p0=[est_k, est_L, est_t0])
        except RuntimeError:
            # Runtime Error occurs when curve fitting takes over maxfev iterations
            # Usually then we have nonsigmoidal data (no response)
            continue
       
        speeds.append(popt[0])
        amplitudes.append(popt[1])
        latencies.append(popt[2])

        if figure_savefn is not None or debug:
            ax.plot(timepoints, displacement, '-')
            ax.plot(timepoints, _logistic_function(timepoints, *popt),
                    '--', label='fit rep {}'.format(i_repeat))
    

    if figure_savefn is not None:
        fig.savefig(figure_savefn)
    if debug:
        plt.show()
        

    return amplitudes, speeds, latencies



def save_sigmoidal_fit_CSV(analysers, savefn, save_fits=True):
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
        writer.writerow(['Name', 'Image folder',  'Mean L (pixels)', 'STD L (pixels)' 'Mean k (pixels/s)', 'STD k (pixels/s)', 'Mean t0 (s)', 'STD t0 (s)', 'Fit image'])
        
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
                amplitudes, speeds, latencies = sigmoidal_fit(analyser, image_folder,
                        figure_savefn=figure_savefn)

                writer.writerow([analyser.folder, image_folder,
                    np.mean(amplitudes), np.std(amplitudes),
                    np.mean(speeds), np.std(speeds),
                    np.mean(latencies), np.std(latencies),
                    os.path.basename(figure_savefn)])
        
                i_fit += 1

    return None

