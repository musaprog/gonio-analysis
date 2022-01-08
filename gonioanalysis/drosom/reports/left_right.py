'''
Exporting results for the set of experiments where
one location in each left/right eye was measured. 
'''
import os
import csv
import numpy as np

from gonioanalysis.drosom.kinematics import (
        mean_max_response,
        _sigmoidal_fit,
        _simple_latencies,
        )
from gonioanalysis.directories import ANALYSES_SAVEDIR

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
            row = []
            for sublist in columns:
                try:
                    row.append(sublist[i])
                except:
                    row.append('')
            writer.writerow(row)
            #writer.writerow([sublist[i] for sublist in columns])


def read_CSV_cols(fn):
    rows = []
    with open(fn, 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            rows.append(row)

    return list(map(list, zip(*rows)))


def left_right_displacements(manalysers, group_name,
        fn_prefix='LR-displacements',
        savedir=LR_SAVEDIR,
        stimuli={'uv': ['uv', ')'], 'green': ['green'], 'NA': []},
        strong_weak_division=False, divide_threshold=3,
        wanted_imagefolders=None,
        microns=True, phase=False, mean_lr=False,
        reference_frame=False):
    '''
    Saves CSV files of left and right eye movements and ERGs.
    
    If many recordings for an eye/stimulus/specimen combination exist,
    then takes the mean of these (so that each eye appears only once).


    Arguments
    ----------
    manalysers : list of objects
        MAnalyser objects for
    group_name : string
        Name that describes the manalysers. For example, "blind_norpa" or
        "controls".
    fn_prefix : string
        Text to append in the beginnign of the CSV filename.
    stimuli : dict of lists of strings
        Each key is the name of the stimulus, and matching value is a list of
        the suffixes that match the stimulus (the suffix in the end of imagefolder name)
    strong_weak_division : bool
        If True, group data based on strong and weak eye instead of
        combined left and right.
    divide_threshold : int
        Related to the strong_weak_divison argument. For some specimen there may be
        recordings only from one eye, and divide_threshold or more is required
        to in total to do the division.
    wanted_imagefolders: None or a dict
        Keys specimen names, items a sequence of wanted imagefolders
        Relaxes horizontal conditions.
    microns : bool
        Convert pixel movement values to microns
    phase : bool
        If True, return phase (vector direction) instead of the magnitude.
    mean_lr : bool
        If True, average the left and right eye data together.
    reference_frame : False or int
        If an interger (between 0 and N_frames-1), use the corresponding
        frame as a reference zero point.

    '''
    
    # each "file" is a list of columns
    
    fs = None
    efs = None
    
    if wanted_imagefolders:
        conditions = [None, None] 
        csv_files = {'NA': []}
    else:
        conditions = [lambda h: h>10, lambda h: h<-10]
        csv_files = {stim: [] for stim in stimuli.keys()}

    for manalyser in manalysers:
        
        # Left eye
        for eye, condition in zip(['left', 'right'], conditions):
            
            if eye=="left" or mean_lr == False:
                eyedata = {stim: [] for stim in stimuli.keys()}

            for image_folder in manalyser.list_imagefolders(horizontal_condition=condition): 
                
                if wanted_imagefolders and image_folder not in wanted_imagefolders.get(manalyser.name, []):
                    # Skip image_folder if not in wanted_imagefolders (if it's not None)
                    continue
                
                if wanted_imagefolders is None:
                    # look for match
                    stims = []
                    for _stim in stimuli.keys():
                        if any([image_folder.endswith(match) for match in stimuli[_stim]]):
                            stims.append( _stim )
                else:
                    stims = ['NA']
                
                for stim in stims:

                    trace = manalyser.get_magnitude_traces(eye,
                            image_folder=image_folder,
                            mean_repeats=True, microns=microns,
                            _phase=phase)
                
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

                        trace = trace[0][0]
                        
                        if reference_frame is not False:
                            trace = [val - trace[reference_frame] for val in trace]

                        eyedata[stim].append(trace)
       
            if eye == "right" or mean_lr == False:
                for stim in stimuli.keys():
                    if eyedata[stim]:
                        column_name = '{}_mean_{}'.format(manalyser.name, eye)

                        csv_files[stim].append( np.mean(eyedata[stim], axis=0).tolist() )
                        csv_files[stim][-1].insert(0, column_name)


        if "ERGs" in manalyser.linked_data:
            data = manalyser.linked_data['ERGs']
            
            repeatdata = {}

            erg_columns = {}
            
            for recording in data:
                name = 'ERGs_' + recording['Stimulus'] +'!;!'+ recording['eye']
                try:
                    N_repeats = int(recording['N_repeats'])
                except ValueError:
                    N_repeats = 1

                if repeatdata.get(name, 0) < N_repeats: 
                    erg_columns[name] = (np.array(recording['data'])-recording['data'][0]).tolist()
                    erg_columns[name].insert(0, '{}_{} (mV)'.format(manalyser.name, recording['eye']))
            
                    if efs is None:
                        efs = recording['fs']
                    elif efs != recording['fs']:
                        raise ValueError('ERGs with multiple sampling frequencies!')

            for name in erg_columns:
                uname = name.split('!;!')[0]
                try:
                    csv_files[uname]
                except:
                    csv_files[uname] = []

                csv_files[uname].append(erg_columns[name])
    
    
    if strong_weak_division:
        new_csv_files = {}
    
        # Process first DPP data then ERGs
        strong_eyes = {}
        keys = [k for k in csv_files if not 'ERGs' in k] + [k for k in csv_files if 'ERGs' in k]        

        for csv_file in keys:
            pairs = []

            column_titles = [column[0] for column in csv_files[csv_file]]

            for column in csv_files[csv_file]:
                
                if not 'right' in column[0]:
                    try:
                        indx = column_titles.index( column[0].replace('left', 'right'))
                    except ValueError:
                        continue

                    pairs.append((column, csv_files[csv_file][indx]))

           
            if len(pairs) > divide_threshold:
                new_csv_files[csv_file+'_strong'] = []
                new_csv_files[csv_file+'_weak'] = []

                for left, right in pairs:
                    # Fixme
                    rdata = [float(num) for num in right[1:]]
                    ldata = [float(num) for num in left[1:]]

                    if not 'ERGs' in csv_file:
                        specimen_name = '_'.join(left[0].split('_')[:-2])
                    else:
                        specimen_name = '_'.join(left[0].split('_')[:-1])
                    
                    print(specimen_name)

                    if 'ERGs' in csv_file:
                        #ab = [400, 800]
                        
                        if strong_eyes[specimen_name] == 'right':
                            new_csv_files[csv_file+'_strong'].append(right)
                            new_csv_files[csv_file+'_weak'].append(left)
                        else:
                            new_csv_files[csv_file+'_strong'].append(left)
                            new_csv_files[csv_file+'_weak'].append(right)
                    else:
                        ab = [None, None]
                        if abs(quantify_metric(rdata, ab=ab)) > abs(quantify_metric(ldata, ab=ab)):
                            new_csv_files[csv_file+'_strong'].append(right)
                            new_csv_files[csv_file+'_weak'].append(left)
                            strong_eyes[specimen_name] = 'right'
                        else:
                            new_csv_files[csv_file+'_strong'].append(left)
                            new_csv_files[csv_file+'_weak'].append(right)
                            strong_eyes[specimen_name] = 'left'

            else:
                new_csv_files[csv_file+'_all'] = csv_files[csv_file]
            
        csv_files = new_csv_files



    os.makedirs(savedir, exist_ok=True)

    for csv_file in csv_files:
        # Mean in the end
        csv_files[csv_file].append(np.mean([csv_files[csv_file][i][1:] for i in range(len(csv_files[csv_file]))], axis=0).tolist())
        try:
            csv_files[csv_file][-1].insert(0, 'mean')
        except AttributeError as e:
            if csv_file.startswith('ERGs'):
                print(csv_files[csv_file])
                raise ValueError("No ERGs, check linking the ERG data")
            else:
                #raise e
                continue

        if csv_file.startswith('ERGs_'):
            ufs = efs
        else:
            ufs = fs

        # Add xaxis (time) in all files
        data = csv_files[csv_file][0][1:]
        xaxis = np.linspace(0, (len(data)-1)/ufs, len(data)).tolist()
        xaxis.insert(0, 'time (s)')
        csv_files[csv_file].insert(0, xaxis)

        fn = '{}_{}_{}.csv'.format(fn_prefix, group_name, csv_file)
        fn = os.path.join(savedir, fn)
        write_CSV_cols(fn, csv_files[csv_file])


def quantify_metric(data1d, metric_type='mean', ab=(None, None)):
    '''
    From a 1D array (time series) quantify single value metric.

    metric_type : string
        "mean" to take the mean of the range
    ab : tuple of integers
        The range as datapoint indices.
    '''
    part = data1d
    if ab[1] is not None:
        part = part[:ab[1]]
    if ab[0] is not None:
        part = part[ab[0]:]

   
    if metric_type == 'mean':
        value = np.mean(part)

    return value


def lrfiles_summarise(lrfiles, point_type='mean', ab=(None, None)):
    '''
    Datapoints for making box/bar plots and/or for statistical testing.

    Arguments
    ---------
    lrfiles : list of filenames
        LR-displacements files of the left_right_displacements.
    point_type : string
        Either "mean" to take mean of the range (used for DPP movement data) or
        min-start to take the mean around the minimum and subtract start value (used for ERGs).
        If not specified, use 'mean'.
    ab : tuple
        Specify the range as indices (rows of the lrfiles excluding the header)
        If not specified, try to autodetect based on if ERGs is contained in the filesames
        ((half,end) for DPP, (400, 800) for ERGs).
    '''
    
    csv_files = {}
    

    for fn in lrfiles:
        
        
        sfn = os.path.basename(fn)
        specimen_name = '_'.join(sfn.split('_')[1:-1])
        stim = sfn.split('_')[-1].split('.')[0]

        csv_rows = csv_files.get(stim, {})

        
        coldata = read_CSV_cols(fn)
        
        if specimen_name not in csv_rows:
            csv_rows[specimen_name] = []
        
        if ab[0] is None or ab[1] is None:
        
            # FIXME Quite specific
            if 'ERGs' in sfn:
                a, b = [400, 800]
            else:
                a, b = [int((len(coldata[0])-1)/2), len(coldata[0])-1]
        else:
            a, b = ab

        # First column is time, the last is the mean, skip these
        for col in coldata[1:-1]:
            
            if point_type != 'kinematics':
                if a is not None and b is not None:
                    numerical_col = [float(num) for num in col[a+1:b]]
                else:
                    numerical_col = [float(num) for num in col[1:]]
            else:
                numerical_col = [float(num) for num in col[1:]]

            if point_type == 'mean':
                value = np.mean(numerical_col)
            elif point_type.startswith('min-start'):
                value = np.min(numerical_col) - float(col[1])
            
            elif point_type == 'kinematics':
                fs = 1/(float(coldata[0][2]) - float(coldata[0][1]))
                value = _sigmoidal_fit([numerical_col], fs)
                
                if value is None:
                    continue

                value = [value[0][0], value[1][0], value[2][0]]
                #value = _simple_latencies([numerical_col], fs)[0]
            
            if len(lrfiles)==1 and point_type == "kinematics":
                # expand CSV rows
                for name, val in zip(['displacement', 'logistic growth rate', '1/2-risetime'], value):
                    try:
                        csv_rows[specimen_name+'_'+name]
                    except:
                        csv_rows[specimen_name+'_'+name] = []
                    csv_rows[specimen_name+'_'+name].append(val)

            else:
                csv_rows[specimen_name].append(value)


        csv_files[stim] = csv_rows

    path = os.path.join(os.path.dirname(lrfiles[0]), 'summary')
    os.makedirs(path, exist_ok=True)

    for fn in csv_files:
        ofn = os.path.join( path, 'LR_summary_{}.csv'.format(fn) )
        with open(ofn, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            for row in csv_files[fn]:
                writer.writerow([row]+csv_files[fn][row])



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


    


