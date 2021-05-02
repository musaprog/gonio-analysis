'''
Linked Biosyst electrophysiology data, intended for electroretinograms (ERGs).
'''

import os
import csv
import glob

import numpy as np

from gonioanalysis.directories import ANALYSES_SAVEDIR

from biosystfiles import extract as bsextract

  
def _load_ergs(ergs_labbook, ergs_rootdir):
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

    csvfile = []
    with open(ergs_labbook, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            csvfile.append(row)
    
    previous_specimen = ''

    # Header (column names)
    column_names = csvfile[0]

    for line in csvfile[1:]:
        efn = line[1]
        match = glob.glob(ergs_rootdir+'/**/'+efn)
        if len(match) != 1:
            print('{} not found'.format(efn))
        else:
            specimen = line[0]
            if not specimen:
                specimen = previous_specimen
            previous_specimen = specimen

            try:
                ergs[specimen]
            except KeyError:
                ergs[specimen] = []

            ddict = {key: value for key, value in zip(column_names[2:], line[2:])}
            trace, fs = bsextract(match[0], 0)

            # Use the mean if many repeats present
            ddict['data'] = np.mean(trace, axis=1).flatten().tolist()
            ddict['fs'] = int(fs)
            ergs[specimen].append(ddict)

    return ergs



def link_erg_labbook(manalysers, ergs_labbook, ergs_rootdir):
    '''
    Links MAnalyser objects with erg data fetched from an electronic
    labbook. For specification, see function _load_ergs
    '''

    erg_data = _load_ergs(ergs_labbook, ergs_rootdir)

    for manalyser in manalysers:
        mname = manalyser.get_specimen_name()
        
        if mname in erg_data:
            manalyser.link_data('ERGs', erg_data[mname])
            manalyser.save_linked_data()

        else:
            print('No ERGs for {}'.format(mname))


