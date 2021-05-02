'''
Analysing and creating figures from the norpA rescues.
'''

import os
import csv
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mati
import matplotlib.patches as patches
import matplotlib.colors as mplcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
#import pandas as pd
#import seaborn as sns
#from statannot import add_stat_annotation

from gonioanalysis.directories import ANALYSES_SAVEDIR
from gonioanalysis.image_tools import open_adjusted


#from xray_erg.plotting import Animator
#from xray_erg.interpolating import interpolate
from biosystfiles import extract as bsextract
#from mpl_steroids.scalebar import add_scalebar


RESCUES = ['Rh1', 'Rh3', 'Rh4', 'Rh5', 'Rh6']
STIMULI = ['uv', 'green', 'nostim']


# FANCY NAMES
STIMULUS_FANCY_NAMES = {STIMULI[0]: 'UV flash', STIMULI[1]: 'Green flash', STIMULI[2]: 'No stimulus',
        STIMULI[0]+'_erg': 'UV flash', STIMULI[1]+'_erg': 'Green flash',
        'sens': ''}


# COLORS
PALETTE = ['violet', 'lime', 'gray']

RESCUE_COLORS = {'Rh1': '#4ce801', 'Rh3': '#e201f0', 'Rh4': '#7e01f1', 'Rh5': '#01d2d3', 'Rh6': '#c4d201'}

STIMULUS_COLORS = {'uv': 'purple', 'green': 'green', 'nostim': 'black'}
STIMULUS_COLORS['uv_erg'] = STIMULUS_COLORS['uv']
STIMULUS_COLORS['green_erg'] = STIMULUS_COLORS['green']


EXPERIMENT_COLORS = {'ergs': (0.99,0.99,0.97), 'dpp': (0.97,0.99,0.99)}


THIS_SAVEDIR = os.path.join(ANALYSES_SAVEDIR, 'norpa_rescues')


# BLockdict for blacklisting
BLOCKDICT = {'norpA_Rh1_02_manyrepeats': 'No ERGs',
        'norpA_Rh1_10_manyrepeats': 'Miniscule ERGs',
        'norpA_Rh4_06_manyrepeats': 'No ERGs',
        'norpA_Rh6_06_manyrepeats': 'ERGs not recorded',
        'norpA_Rh1_06_manyrepeats_right': 'Not responding to Green',
        'norpA_Rh3_01_manyrepeats_left': 'Not clearly responding to UV',
        'norpA_Rh3_03_manyrepeats_right': 'Not clearly responding to UV',
        'norpA_Rh3_04_manyrepeats_right': 'Not clearly responding to UV',
        'norpA_Rh3_07_manyrepeats_right': 'Not clearly responding to UV',
        'norpA_Rh5_03_manyrepeats_right': 'Not clearly responding to UV',
        'norpA_Rh5_05_manyrepeats_right': 'Not clearly responding to UV',
        'norpA_Rh5_09_manyrepeats_left': 'Not clearly responding to UV'
        }

# Temporal pixel size, should go to analysing.py
PXLSIZE=0.81741 # microns per pixel

# Led spectrums
LED_SPECTRUM_DIR = '/home/joni/data/DPP/DPP_cal_spectrometer_data/DPP_cal_10/'
LED_SPECTRUM_FNS = ['green_center_8V_calculated.csv', 'uv_center_8V.csv', 'ergsetup_green_center_5V.csv', 'ergsetup_uv_center_5V.csv'] 
LED_SPECTRUM_FNS = [os.path.join(LED_SPECTRUM_DIR, fn) for fn in LED_SPECTRUM_FNS]


def norpa_rescue_manyrepeats(manalysers):
    '''
    Analyse norpA Rh{1,3,4,5,6} rescue mutants recorded with
    experimental protocol manyrepeats.

    In the so called manyrepeats protocol, each eye was imaged
    with 25 repeats while flashing green, UV or no stimulus
    at location vertical= -37 deg and horizontal= +- 28 deg.
    
    The experiment was rotated almost without exception as follows:
        1) right eye UV
        2) right eye NOSTIM
        3) right eye GREEN
        4) left eye GREEN
        5) left eye NOSTIM
        5) left eye UV

    Specimens were named as norpA_Rh{i_rhodopsin}_{i_specimen}_manyrepeats,
    where i_rhodopsin is 1, 3, 4, 5, or 6 and i_specimen 01, 02, ...
    '''

    results = {}
    
    rescues = RESCUES
    stimuli = STIMULI

    
    

    for manalyser in manalysers:
        
        specimen_name = manalyser.get_specimen_name()
        
        if specimen_name in BLOCKDICT.keys():
            print('Specimen {} on the block list because {}'.format(specimen_name, BLOCKDICT[specimen_name]))
            continue

        # Look which mutant we have
        specimen_rescue = None
        for rescue in rescues:
            if rescue in specimen_name:
    
                if specimen_rescue is not None:
                    raise ValueError('2 or more Rhi rescues fit the name {}'.format(specimen_name))
                
                specimen_rescue = rescue
                break
        
        # If none of the rescues, then skip this Manalyser
        if specimen_rescue is None:
            continue


        # Then iterate over the image folder
        for image_folder in sorted(manalyser.list_imagefolders(list_special=False)):

            # Those image folders without suffix were in my recordings always
            # actually stimulated with UV; It was just in the beginning, I
            # sometimes forgot to add the suffix UV. Recordings were always
            # started with UV.
            specimen_stimtype = 'uv'

            for stimtype in stimuli:
                if image_folder.endswith('_'+stimtype):
                    specimen_stimtype = stimtype

            data = manalyser.get_movements_from_folder(image_folder)
            eye = list(data.keys())[0]
            data = data[eye]

            print('Specimen {}_{}'.format(specimen_name, eye))
            if '{}_{}'.format(specimen_name, eye) in BLOCKDICT.keys():
                print('  Specimen {}_{} blocked'.format(specimen_name, eye))
                continue

            times = []
            traces = []
            responses = []

            for i_repeat in range(len(data)):
                X = np.asarray(data[i_repeat]['x'])
                Y = np.asarray(data[i_repeat]['y'])

                mag_trace = np.sqrt(X**2+Y**2)
                
                A = np.mean(mag_trace[:4])
                B = np.mean(mag_trace[-10:])
                response = B-A
            
                traces.append(mag_trace)
                responses.append(response)
                times.append(manalyser.get_recording_time(image_folder, i_rep=i_repeat))


            if specimen_rescue not in results.keys():
                results[specimen_rescue] = {}

            if specimen_stimtype not in results[specimen_rescue].keys():
                results[specimen_rescue][specimen_stimtype] = {'times': [], 'separate_traces': [], 'traces': [],
                        'sexes': [], 'names': [], 'ages': [], 'responses': [], 'eyes': [], 'image_interval': []}

            results[specimen_rescue][specimen_stimtype]['separate_traces'].append(traces)
            results[specimen_rescue][specimen_stimtype]['traces'].append(np.mean(traces, axis=0))
            results[specimen_rescue][specimen_stimtype]['responses'].append(responses)

            results[specimen_rescue][specimen_stimtype]['times'].append( times[int(len(times)/2)] )
            results[specimen_rescue][specimen_stimtype]['ages'].append( manalyser.get_specimen_age() )
            results[specimen_rescue][specimen_stimtype]['sexes'].append( manalyser.get_specimen_sex() )
            results[specimen_rescue][specimen_stimtype]['names'].append( manalyser.get_specimen_name() )
            results[specimen_rescue][specimen_stimtype]['eyes'].append( eye )
            results[specimen_rescue][specimen_stimtype]['image_interval'].append( manalyser.get_image_interval() ) 
        

    #plot_individual(manalysers, results)
    #plot_mean_timecourses(results)
    #plot_mean_timecourses(results, sex='male')
    #plot_box_summary(results)    

    export_mean_timecourses(results)


def plot_bar_summary(results):
    

    panda_data = []
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            responses = results[rescue][stimtype]['responses']
            #for i_repeat, value in enumerate(values):
            panda_data.append([rescue, STIMULUS_FANCY_NAMES[stimtype], np.mean(responses), np.std(responses), responses])

    df = pd.DataFrame(panda_data, columns=['norpA rescue', 'Stimulus type', 'mean', 'std', 'responses'])
    print(df)
    

    a = df.pivot('norpA rescue', 'Stimulus type', 'mean').plot(kind='bar',
            yerr=df.pivot('norpA rescue', 'Stimulus type', 'std'))

def plot_box_summary(results):
    
    rescue_labels = {'No s'}

    panda_data = []
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            data = results[rescue][stimtype]
            for i_eye in range(len(data['eyes'])):
                panda_data.append([rescue, STIMULUS_FANCY_NAMES[stimtype],
                    data['names'][i_eye], data['eyes'][i_eye], np.mean(data['responses'][i_eye])])

    df = pd.DataFrame(panda_data, columns=['norpA rescue', 'Stimulus type', 'name', 'eye', 'response'])
    print(df)
    

    #a = df.pivot('norpA rescue', 'Stimulus type', 'mean').plot(kind='bar',
    #        yerr=df.pivot('norpA rescue', 'Stimulus type', 'std'))
    
    plt.figure()
    ax = sns.boxplot(x='norpA rescue', y='response', hue='Stimulus type', data=df,
            hue_order=[STIMULUS_FANCY_NAMES[stim] for stim in STIMULI], palette=PALETTE)
    
    box_pairs = []
    for rescue in results.keys():
        for i in range(len(STIMULI)-1):
            box_pairs.append(((rescue, STIMULUS_FANCY_NAMES[STIMULI[i]]), (rescue, STIMULUS_FANCY_NAMES[STIMULI[-1]])))
    
    print(box_pairs)

    add_stat_annotation(ax, data=df, x='norpA rescue', y='response', hue='Stimulus type',
            box_pairs=box_pairs, test='Wilcoxon', loc='inside')
    #ax = sns.swarmplot(x="norpA rescue", y="response", data=df, color=".25")

def plot_mean_timecourses(results, sex=None):
    '''
    Create a figure with time courses.
    '''


    fig = plt.figure(figsize=(16,9))
    subplots = {}
    
    nrows = 5
    ncols = 4

    cols = ['', 'sens', 'uv', 'green', 'nostim', 'uv_erg', 'green_erg']
    rows = ['colnames', 'Rh1', 'Rh3', 'Rh4', 'Rh5', 'Rh6']
    

    maxval = 12.5
   

    animator = Animator(fig)


    for row in rows[::-1]:
        for col in cols:
            irow = rows.index(row)
            icol = cols.index(col)
            
            ax = fig.add_subplot(len(rows),len(cols), 1+len(cols)*irow+icol)
            
            if row == rows[-1] and col == 'sens':
                ax.set_xlabel('Wavelength (nm)')
                ax.get_yaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
            else:
                ax.set_axis_off()

            if col == 'sens' or row == 'colnames':
                ax.set_xlim(300,650)
                ax.set_ylim(0,1.05)

            subplots[row+col] = ax

            if icol == 1 and row!='colnames':
                color = RESCUE_COLORS.get(row, 'black')
                ax.text(0, 0.5, row, va='center', ha='right',
                        color=color, transform=ax.transAxes)
            
            if irow == 0:
                color = STIMULUS_COLORS.get(col, 'black')
                ax.text(1-0.618, 0.618, STIMULUS_FANCY_NAMES.get(col, col), va='bottom', ha='center',
                        color=color, transform=ax.transAxes)
    
    # Set titles ERG and DPP
    subplots['colnames'+'green'].text(0.5, 1.3, 'DPP microsaccades',
            ha='center', va='bottom', fontsize='large',
            backgroundcolor=EXPERIMENT_COLORS['dpp'], transform=subplots['colnames'+'green'].transAxes)

    subplots['colnames'+'green_erg'].text(0, 1.3, 'ERGs',
            ha='center', va='bottom', fontsize='large',
            backgroundcolor=EXPERIMENT_COLORS['ergs'], transform=subplots['colnames'+'green_erg'].transAxes)
    
    # Title lines
    #box1 = subplots['colnames'+'uv'].get_position()
    #box2 = subplots['colnames'+'nostim'].get_position()
    #arrow = patches.FancyArrow(box1.x0, box1.y1+box1.height/10, box2.x1-box1.x0, 0,
    #        shape='right', hatch='|', transform=fig.transFigure, figure=fig)
    #fig.patches.extend([arrow])

    
    # Plot norpA rescue illustrative images
    imagedir = '/home/joni/Pictures/NorpA rescues/'
    for fn in os.listdir(imagedir):
        for row in rows:
            if row in fn:
                image = mati.imread(os.path.join(imagedir, fn))
                h,w,d = image.shape
                
                im = subplots[row+''].imshow(image)
                
                im.set_clip_path( patches.Circle((int(w/2),int(h/2)),int(min(w,h)/2), transform=subplots[row].transData) )
                continue
    
    led_wavelengths, led_spectrums = _load_led_spectrums(LED_SPECTRUM_FNS)
    
    # Plot spectral sensitivities
    wave_axis, sensitivities = _load_spectral_sensitivities()
    for rescue, sensitivity in zip([row for row in rows if row in RESCUES], sensitivities):
        subplots[rescue+'sens'].plot(wave_axis, sensitivity, color=RESCUE_COLORS[rescue])
        
        # Plot LED spectrums in each figure
        subplots[rescue+'sens'].plot(led_wavelengths[0], led_spectrums[0], '--', color='green', lw=1)
        subplots[rescue+'sens'].plot(led_wavelengths[0], led_spectrums[1], '--', color='purple', lw=1)
    

    # Plot stimulus/LED spectral curves
    subplots['colnames'+'green'].plot(led_wavelengths[0], led_spectrums[0], color='green')
    subplots['colnames'+'uv'].plot(led_wavelengths[0], led_spectrums[1], color='purple')
    subplots['colnames'+'nostim'].plot([led_wavelengths[0][0], led_wavelengths[0][-1]], [0,0], color='black')
    subplots['colnames'+'green_erg'].plot(led_wavelengths[0], led_spectrums[2], color='green')
    subplots['colnames'+'uv_erg'].plot(led_wavelengths[0], led_spectrums[3], color='purple')
 
    # Plot ERGs

    erg_data = _load_ergs()
    
    ergs_min = np.inf
    ergs_max = -np.inf
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            
            erg_traces = []
            
            # For this rescue/stimtype combination, go through every erg
            for specimen_name in erg_data.keys():
                if specimen_name in BLOCKDICT.keys():
                    print('Blocked {}'.format(specimen_name))
                    continue
                # Looking for correct specimen
                if rescue in specimen_name:
                    # and for
                    for ergs in erg_data[specimen_name]:
                        # correct stimulus type
                        if stimtype in ergs and '25' in ergs:
                            erg = ergs[0][0]
                            erg = erg - erg[0]

                            color = np.array(mplcolors.to_rgb(RESCUE_COLORS[rescue]))
                            color = np.mean([color, (1,1,1)], axis=0)
                            subplots[rescue+stimtype+'_erg'].plot(erg, color=color, lw=1)
                            erg_traces.append(erg)
                            
                            ergs_min = min(ergs_min, np.min(erg))
                            ergs_max = max(ergs_max, np.max(erg))

            if erg_traces:
                color = np.array(mplcolors.to_rgb(RESCUE_COLORS[rescue])) * 0.75
                subplots[rescue+stimtype+'_erg'].plot(np.mean(erg_traces, axis=0), color=color)
    
    # Set ERG axis limits
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
                if rescue+stimtype+'_erg' in subplots.keys():
                    subplots[rescue+stimtype+'_erg'].set_ylim(ergs_min, ergs_max)

    
    # Set DPP and ERG plots background color

    box1 = subplots['Rh6'+'uv_erg'].get_position()
    box2 = subplots['Rh1'+'green_erg'].get_position()
    rect = patches.Rectangle((box1.x0, box1.y0), (box2.x0-box1.x0)+box2.width, (box2.y0-box1.y0)+box2.height, color=EXPERIMENT_COLORS['ergs'], zorder=-1)
    fig.add_artist(rect)         

    box1 = subplots['Rh6'+'uv'].get_position()
    box2 = subplots['Rh1'+'nostim'].get_position()
    rect = patches.Rectangle((box1.x0, box1.y0), (box2.x0-box1.x0)+box2.width, (box2.y0-box1.y0)+box2.height, color=EXPERIMENT_COLORS['dpp'], zorder=-1)
    fig.add_artist(rect)         
    
    
    # Add scale bars
    add_scalebar(subplots['Rh6'+'nostim'], 50, 5, position=(1,5), xunits='ms', yunits='µm')
    add_scalebar(subplots['Rh6'+'green_erg'], 300, -3, position=(100,-3), xunits='ms', yunits='mV')
    

    # Plot DPP data
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            
            ax = subplots[rescue+stimtype]

            traces = results[rescue][stimtype]['traces']
            
            # How many fps images were taken by camera
            image_interval = results[rescue][stimtype]['image_interval']

            color = None
            for i in range(len(traces)):
                if sex is not None:
                    print(results[rescue][stimtype]['sexes'][i])
                    if sex == results[rescue][stimtype]['sexes'][i]:
                        color = 'red'
                    else:
                        color = 'gray'

                x, y = interpolate(np.arange(0, len(traces[i])*image_interval[i]*1000, image_interval[i]*1000), traces[i]*PXLSIZE, len(traces[i]))
                color = np.array(mplcolors.to_rgb(RESCUE_COLORS[rescue]))
                color = np.mean([color, (1,1,1)], axis=0)               
                line = ax.plot(x, y, label=results[rescue][stimtype]['ages'][i], lw=1, color=color)
                
                animator.add_animation(line[0], [x,y], hide=False)

            color = np.array(mplcolors.to_rgb(RESCUE_COLORS[rescue])) * 0.75
            ax.plot(x, np.mean(traces,axis=0)*PXLSIZE, label='mean', lw=2, color=color)

            #ax.set_title(rescue+'_'+stimtype)
            ax.set_ylim(0,maxval)

    
    animator.frames += len(x)
    
    os.makedirs(THIS_SAVEDIR, exist_ok=True)
    
    animation = animator.get_animation(interval=40)
    #animation.save(os.path.join(THIS_SAVEDIR, 'timecourses.mp4'), dpi=600)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)



def export_mean_timecourses(results):
    '''
    Exports the plot_mean_timecourses as csv files for plotting/manipulating
    with an external program.
    '''

    savedir = os.path.join(THIS_SAVEDIR, 'exports')
    os.makedirs(savedir, exist_ok=True)

    # Each item is for a file
    csv_files = {}
    
    # DPP data
    for rescue in results.keys():
        for stimtype in results[rescue].keys():

            # Each item for column
            csv_file = {}

            traces = results[rescue][stimtype]['traces']
            image_interval = results[rescue][stimtype]['image_interval']
            
            avg_traces = []

            for i in range(len(traces)):
                x, y = interpolate(np.arange(0, len(traces[i])*image_interval[i]*1000, image_interval[i]*1000), traces[i]*PXLSIZE, len(traces[i]))
                
                if i == 0:
                    csv_file['time (ms)'] = x
                
                specimen_name = results[rescue][stimtype]['names'][i]
                eye = results[rescue][stimtype]['eyes'][i]
                column_name = '{}_mean_{}'.format(specimen_name, eye)

                csv_file[column_name] = y

                avg_traces.append(y)
            
    
            if csv_file:
                # Add mean trace
                csv_file['mean'] = np.mean(avg_traces, axis=0)
                csv_files[rescue+'_'+stimtype] = csv_file

    # ERGs
    erg_data = _load_ergs()
    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            
            csv_file = {}

            traces = []
            
            # For this rescue/stimtype combination, go through every erg
            for specimen_name in erg_data.keys():
                if specimen_name in BLOCKDICT.keys():
                    print('Blocked {}'.format(specimen_name))
                    continue
                # Looking for correct specimen
                if rescue in specimen_name:
                    # and for
                    for ergs in erg_data[specimen_name]:
                        # correct stimulus type
                        if stimtype in ergs and '25' in ergs:
                            erg = ergs[0][0]
                            erg = erg - erg[0]
                            erg = erg.flatten()
                            
                            if not csv_file:
                                fs = ergs[0][1]
                                dt = 1/fs
                                csv_file['time (ms)'] = 1000 * np.arange(0, len(erg)*dt, dt)
                            
                            csv_file[specimen_name+'_erg'] = erg
                            traces.append(erg)
            if csv_file:
                csv_file['mean'] = np.mean(traces, axis=0)
                csv_files[rescue+'_'+stimtype +'_erg'] = csv_file

    # Measured specturms for LEDs
    led_wavelengths, led_spectrums = _load_led_spectrums(LED_SPECTRUM_FNS)
    
    for i_led, (wave, spec) in enumerate(zip(led_wavelengths, led_spectrums)):
        csv_file = {}
        csv_file['wavelength (nm)'] = wave
        csv_file['relative_intensity'] = spec
        csv_files[os.path.basename(LED_SPECTRUM_FNS[i_led]).rstrip('.csv')] = csv_file
        print('i_led {}'.format(i_led))
    
    # Spectral sensitivities
    wave_axis, sensitivities = _load_spectral_sensitivities()
    for rescue, sensitivity in zip(RESCUES, sensitivities):
        
        csv_file = {}
        csv_file['wavelength'] = wave_axis 
        csv_file['sensitivity'] = sensitivity

        csv_files['sensitivity_{}'.format(rescue)] = csv_file
    

    # Export
    for csv_name, csv_file in csv_files.items():
        
        with open(os.path.join(savedir, csv_name+'.csv'), 'w') as fp:
            
            writer = csv.writer(fp)

            column_names = sorted(list(csv_file.keys()))
            if 'time (ms)' in column_names:
                column_names.insert(0, column_names.pop(column_names.index('time (ms)')))
            
            if 'mean' in column_names:
                column_names.insert(len(column_names), column_names.pop(column_names.index('mean')))
            
            
            N = len(csv_file[column_names[0]])
            
            print("{} len {}".format(csv_name, len(column_names)-2))
            
            writer.writerow(column_names)

            for i in range(0, N):
                row = [csv_file[col][i] for col in column_names]
                
                writer.writerow(row)




def plot_ergs():
    '''
    Plot ERGs alone.
    '''

    ergs = get_ergs()
    
    for rescue in RESCUES:
        for stimtype in STIMULI:
            pass


    
def _load_ergs():
    '''
    Fetches ERGs for the specimen matching the name.

    Requirements
    - ERGs are Biosyst recorded .mat files.
    - Requires also a lab book that links each specimen name to a ERG file,
        and possible other parameter values such as intensity, repeats,
        UV/green etc.

    Returns ergs {specimen_name: data}
        where data is a list [[ergs, par1, par2, ...],[..],..]
        ergs are np arrays
    '''
    ergs = {}

    ergs_rootdir = '/home/joni/data/DPP_ERGs'
    ergs_labbook = '/home/joni/data/DPP_ERGs/labbook_norpa_rescues.csv'
    
    csvfile = []
    with open(ergs_labbook, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            csvfile.append(row)
    
    previous_specimen = ''

    for line in csvfile:
        efn = line[1]
        match = glob.glob(ergs_rootdir+'/**/'+efn)
        if len(match) != 1:
            print('{} not found'.format(efn))
            #ergs.append(None)
        else:
            #print(efn + ' ' + match[0])

            specimen = line[0]
            if not specimen:
                specimen = previous_specimen
            previous_specimen = specimen

            try:
                ergs[specimen]
            except KeyError:
                ergs[specimen] = []

            ergs[specimen].append([bsextract(match[0], 0), *line[2:]])

    return ergs


def _load_spectral_sensitivities(fn='/home/joni/analyses/digitized_rh_rescues.csv'):
    '''
    Spectral sensitivies of different Rh opsins. Digitalized from a figure into a csv file.

    Returns
        wavelengths, [rh1, ...]
        where wavelengths a numpy array 1D
        and rh1, rh3,... also
    '''

    data = np.loadtxt(fn, delimiter=' ', skiprows=1)
    print(data.shape) 
    return data[:,0], [data[:, i] for i in range(1, data.shape[1])]



def _load_led_spectrums(spectrometer_csv_files):
    '''
    Returns spectrums.
    
    spectometer_csv_files       A list of spectrometer csv files.

    Returns     wavelengts  spectrums
        which both are 1d lists containing 1d numpy arrays
    '''
    
    wavelengths = []
    spectrums = []
    
    # Load spectrums
    for fn in spectrometer_csv_files:
        spectrum = np.loadtxt(fn, delimiter=',', skiprows=1)[:,1]
        
        wavelength = np.loadtxt(fn, delimiter=',', skiprows=1)[:,0]
        wavelength = [207.1545+0.3796126*w+0.00002822671*(w**2) for w in wavelength]
        
        spectrums.append(spectrum)
        wavelengths.append(wavelength)

    
    # Load integration times from txt files
    for i, fn in enumerate([fn.rstrip('.csv')+'.txt' for fn in spectrometer_csv_files]):
        with open(fn, 'r') as fp:
            integration_time = fp.readline()
        integration_time = float(integration_time.split(' ')[-2]) # ms
        
        spectrums[i] = spectrums[i] / integration_time

    cmax = np.max(spectrums)

    return wavelengths, [spectrum/cmax for spectrum in spectrums]




def _get_results_keys(results, specimen_name):
    '''
    '''
    
    matches = []

    for rescue in results.keys():
        for stimtype in results[rescue].keys():
            try:
                index = results[rescue][stimtype]['names'].index(specimen_name)
                matches.append([rescue, stimtype, index])
            except:
                pass

    return matches


def plot_individual(manalysers, results):
    
    

    pdf_savedir = os.path.join(ANALYSES_SAVEDIR, 'norpa_rescues')
    os.makedirs(pdf_savedir, exist_ok=True)

    ergs = _load_ergs()
    
    
    with PdfPages(os.path.join(pdf_savedir, 'individual.pdf')) as pdf:

        for manalyser in manalysers[:5]:
            specimen_name = manalyser.get_specimen_name()
            try:
                ergs[specimen_name]
                print('ERGS')
            except KeyError:
                print('No ERGs for {}'.format(specimen_name))
            
            keys = _get_results_keys(results, specimen_name)
            print(keys)

            # Specimen info
            fig, axes = plt.subplots(1+len(keys),2, figsize=(8.3,11.7))
            plt.title(specimen_name)
            
            # Plot face image
            axes[0][0].imshow(open_adjusted(manalyser.get_snap()), cmap='gray')

            # Plot DPP

            for i, key in enumerate(keys):
                axes[i][0].plot( results[key[0]][key[1]]['separate_traces'][key[2]] )
            

            pdf.savefig()

            plt.close()
	




 

