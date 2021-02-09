'''
Exporting results for the set of experiments where
one location in each left/right eye was measured. 
'''
import os
import csv
import numpy as np

from pupilanalysis.drosom.kinematics import mean_max_response
from pupilanalysis.directories import ANALYSES_SAVEDIR

LR_SAVEDIR = os.path.join(ANALYSES_SAVEDIR, 'LR_exports')

def write_CSV_cols(fn, columns):
    '''
    Note
    -----
    Writes as many rows as there are rows in the first columns.

    Attributes
    ----------
    columns : list of lists
        Each item is column, and in each item that is also a list,
        each item is the value of the row.
    '''


    with open(fn, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        for i in range(len(columns[0])):
            writer.writerow([sublist[i] for sublist in columns])



def left_right_displacements(manalysers, group_name,
        fn_prefix='LR-displacements',
        savedir=LR_SAVEDIR,
        stimuli={'uv': 'uv', 'green': 'green'}):
    '''
    

    Arguments
    ----------
    manalysers : list of objects
        MAnalyser objects for
    group_name : string
        Name that describes the manalysers. For example, "blind_norpa" or
        "controls".
    fn_prefix : string
        Text to append in the beginnign of the CSV filename.
    stimuli : dict of strings
        Each key is the name of the stimulus, and matching value is the suffix
        that matches the stimulus (the suffix in the end of imagefolder name)
    '''
    
    # each "file" is a list of columns
    csv_files = {stim: [] for stim in stimuli.keys()}
    
    fs = None
    
    for manalyser in manalysers:
        
        # Left eye
        for eye, condition in zip(['left', 'right'], [lambda h: h>20, lambda h: h<-20]):
            for image_folder in manalyser.list_imagefolders(horizontal_condition=condition): 
                
                for stim in stimuli.keys():
                    if image_folder.endswith(stimuli[stim]):
                        
                        trace = manalyser.get_magnitude_traces(eye,
                                image_folder=image_folder,
                                mean_repeats=True)
                    
                        trace = list(trace.values())
                        if len(trace) >= 2:
                            raise NotImplementedError('mistake in implementation')
                        

                        if trace:
                            # Check that fs matches
                            nfs = manalyser.get_imaging_frequency(image_folder=image_folder)
                            if fs is None:
                                fs = nfs
                            elif fs != nfs:
                                raise ValueError('Analysers with multiple fs!')

                            column_name = '{}_mean_{}'.format(manalyser.name, eye)
                            
                            trace = trace[0][0].tolist()
                            trace.insert(0, column_name)
                            csv_files[stim].append(trace)

    os.makedirs(savedir, exist_ok=True)

    for csv_file in csv_files:
        
        # Add xaxis (time) in all files
        data = csv_files[csv_file][0][1:]
        xaxis = np.linspace(0, (len(data)-1)/fs, len(data)).tolist()
        xaxis.insert(0, 'time (s)')
        print(xaxis)
        csv_files[csv_file].insert(0, xaxis)

        fn = '{}_{}_{}.csv'.format(fn_prefix, group_name, csv_file)
        fn = os.path.join(savedir, fn)
        write_CSV_cols(fn, csv_files[csv_file])


def left_right_ergs(manalysers):
    '''
    
    '''
    pass

def left_right_summary(manalysers):
    '''
    Condensed from left/right
    '''
    csvfile = []
    
    header = ['Specimen name', 'Left mean response (pixels)', 'Right mean response (pixels)', 'Right eye ERG response (mV)', 'Eyes with microsaccades', '"Normal" ERGs']
    
    csvfile.append(header)

    for manalyser in manalysers:
        
        print(manalyser.get_specimen_name())

        line = []

        line.append(manalyser.get_specimen_name())
        
        # Left eye

        responses = []
        
        for image_folder in manalyser.list_imagefolders(horizontal_condition=lambda h: h>=0): 
            if image_folder.endswith("green"):
                continue
            
            print(image_folder)
            mean = mean_max_response(manalyser, image_folder)
            if not np.isnan(mean):
                responses.append(mean)
        
        line.append(np.mean(responses))
        
        # Right eye
        responses = []
        
        for image_folder in manalyser.list_imagefolders(horizontal_condition=lambda h: h<0): 
            if image_folder.endswith('green'):
                continue
            mean = mean_max_response(manalyser, image_folder)
            if not np.isnan(mean):
                responses.append(mean)
        
        line.append(np.mean(responses))
        
   
        uvresp = None
        greenresp = None
        
        print(manalyser.linked_data.keys())

        if "ERGs" in manalyser.linked_data:
            data = manalyser.linked_data['ERGs']
            
            print(data[0].keys())

            for recording in data:
                print(recording.keys())
                if int(recording['N_repeats']) == 25:

                    if recording['Stimulus'] == 'uv':
                        uvresp = np.array(recording['data'])
                    if recording['Stimulus'] == 'green':
                        greenresp = np.array(recording['data'])

        

        uvresp = uvresp - uvresp[0]
        uvresp = np.mean(uvresp[500:1000])
        
        greenresp = greenresp - greenresp[0]
        greenresp = np.mean(greenresp[500:1000])
        
        line.append(uvresp)
        
        # Descide groups
        dpp_threshold = 0.9999
        if line[1] < dpp_threshold and line[2] < dpp_threshold:
            line.append('None')
        elif line[1] < dpp_threshold or line[2] < dpp_threshold:
            line.append('One')
        else:
            line.append('Two')

        erg_threshold = 0.5
        if abs(line[3]) < erg_threshold:
            line.append('No')
        else:
            line.append('Yes')


        csvfile.append(line)


    for line in csvfile:
        print(line)
    
    with open("left_right_summary.csv", 'w') as fp:
        writer = csv.writer(fp)
        for line in csvfile:
            writer.writerow(line)


    


