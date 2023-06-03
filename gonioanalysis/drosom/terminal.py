#!/usr/bin/env python3
'''Command line interface for GonioAnalysis motion measurement.
'''

import sys
import os
import argparse

import matplotlib.pyplot as plt

from gonioanalysis.drosom.analyser_commands import (
        ANALYSER_CMDS,
        DUALANALYSER_CMDS,
        MULTIANALYSER_CMDS,
        )
from gonioanalysis.droso import DrosoSelect
# Import analysers
from gonioanalysis.drosom.analysing import MAnalyser, MAverager
from gonioanalysis.drosom.orientation_analysis import OAnalyser
from gonioanalysis.drosom.optic_flow import FAnalyser
from gonioanalysis.drosom.transmittance_analysis import TAnalyser

# Avoid importing tkinter bits if not needed
if '--tk_waiting_window' in sys.argv:
    from gonioanalysis.tkgui.widgets import WaitingWindow



Analysers = {'orientation': OAnalyser, 'motion': MAnalyser, 'flow': FAnalyser,
        'transmittance': TAnalyser}

analyses = {**ANALYSER_CMDS, **DUALANALYSER_CMDS, **MULTIANALYSER_CMDS}



def parse_key_value_pairs(string, splitchar=','):
    return {opt.split('=')[0]: opt.split('=')[1] for opt in string.split(splitchar)}


def convert_kwarg(key, value):
    '''Take good guesses to converte kwargs to right type for analyser_cmds.

    Originally all items are strings.
    '''
    if key in ['elev', 'azim', 'dpi']:
        value = float(value)
    elif value.lower()  == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    elif ',' in value:
        value = value.split(',')
        if all([val.replace('.', '', 1).removeprefix('-').isdigit() for val in value]):
            value = [float(val) for val in value]
    return value


def main(custom_args=None):
    
    if custom_args is None:
        custom_args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    # DATA INPUT ARGUMENTS
    parser.add_argument('-D', '--data_directory', nargs='+',
            help='Data directory')

    parser.add_argument('-S', '--specimens', nargs='+',
            help=('Comma separeted list of specimen names.'
                ' Separate groups by space when averaging is on.'
                ' If needed, wanted image folders can be specified with semicolons'
                ' for example, specimen1;imf1;imf2:specimen2;imf1'
                ' (does not work with groups). If image folders contain commas, use'
                ' semicolons, use colons (:) to separate specimens'))
    

    # SETTINGS OR ACTIONS FOR ANALYSERS
    parser.add_argument('-a', '--average', action='store_true',
            help='Average and interpolate the results over the specimens')
    
    parser.add_argument('-t', '--type', nargs='+',
            help='Analyser type, either "motion" or "orientation". Space separate gor groups')
   
    parser.add_argument(
            '--analyser-options', nargs='+',
            help='Extra arguments to ui options. Space separate for groups')

    parser.add_argument('-r', '--reselect-rois', action='store_true',
            help='Reselect ROIs')
 
    parser.add_argument('-R', '--recalculate-movements', action='store_true',
            help='Recalculate with Movemeter')

    parser.add_argument('--active-analysis', nargs='?', default='',
            help="Name of the analyser's active analysis (nothing to do with the --analysis/-A option)")

    # OTHER SETTINGS
    parser.add_argument('--tk_waiting_window', action='store_true',
            help='(internal) Launches a tkinter waiting window')
    parser.add_argument('--dont-show', action='store_true',
            help='Skips showing the plots')
    parser.add_argument('--savefig', action='store_true',
            help='Saves the matplotlib figure if the analysis it produces')

    # Different analyses for separate specimens

    parser.add_argument('-A', '--analysis', nargs=1,
            choices=analyses.keys(),
            help='The performed analysis or action')
   
    parser.add_argument('--analysis-options', nargs='+',
            help='Keyword arguments to the analysis function')

    # Other settings
    parser.add_argument('-o', '--output',
            help='Output filename for export analysis options')


    args = parser.parse_args(custom_args)
    

    if args.tk_waiting_window:
        waiting_window = WaitingWindow('terminal.py', 'When ready, this window closes.')


    # Getting the data directory
    # --------------------------
    if args.data_directory:
        print('Using data directory {}'.format(args.data_directory[0]))
        
        data_directories = args.data_directory
    else:
        # If all FAnalysers, data directoy not needed
        if isinstance(args.type, list) and all([t=='flow' for t in args.type]):
            data_directories = []
        else:
            data_directories = input('Data directory >> ')
    
    # Check that the data directory exists
    for directory in data_directories:
        if not os.path.isdir(directory):
            raise ValueError("{} is not a directory".format(directory))

    # {specimen name : [image_folder_1, ...]}
    wanted_imagefolders = {}

    # Getting the specimens
    # ---------------------
    directory_groups = []
    if args.specimens:
        
        for i_group, group in enumerate(args.specimens):
            print('Using specimens {}'.format(group))
            
            # For flow analysers give fake folders
            if isinstance(args.type, list) and args.type[i_group] == 'flow':
                directory_groups.append( ['none/none'] )
                continue
                
            if group == 'none':
                directory_groups.append(None)
                continue

            if ':' in group:
                specimen_separator = ':'
            else:
                specimen_separator = ','

            if ';' in group:
                
                for specimen in group.split(specimen_separator):
                    if ';' in specimen:
                        splitted = specimen.split(';')
                        wanted_imagefolders[splitted[0]] = splitted[1:]
                
                # Remove specified imagefolders
                group = ','.join([z.split(';')[0] for z in group.split(specimen_separator)])
            
            
            # dont commit me
            group = group.replace(':', ',')
            
            
            directories = []
            for directory in data_directories:
                selector = DrosoSelect(datadir=directory)
                directories.extend( selector.parse_specimens(group) )
            directory_groups.append(directories)
    else:
        selector = DrosoSelect(datadir=data_directories[0])
        directories = selector.ask_user()
     
            
    # Setting up analysers
    # ---------------------
    
    if not args.type:
        args.type = ['motion' for i in directory_groups]

    analyser_groups = []
    
    for i_group, directories in enumerate(directory_groups):

        analysers = []
        Analyser = Analysers[args.type[i_group]]
        
        print('Using {}'.format(Analyser.__name__))

        if directories is None:
            analysers.append(Analyser(None, None))
        else:

            for directory in directories: 
                
                path, folder_name = os.path.split(directory)
                analyser = Analyser(path, folder_name) 
                
                if args.active_analysis:
                    analyser.active_analysis = args.active_analysis
                
                analysers.append(analyser)
         
        if args.analyser_options:
            print(f'  Setting options {args.analyser_options[i_group]}')
            for analyser in analysers:
                # Special empty ones to sign that for analysers in i_group
                # we don't want to set any options
                if args.analyser_options[i_group].lower() in ['na', 'none']:
                    continue
                # Parse analyser options
                opts = parse_key_value_pairs(args.analyser_options[i_group]) 
                analyser.set_ui_options(opts)

        # Ask ROIs if not selected
        for analyser in analysers:
            if analyser.are_rois_selected() == False or args.reselect_rois:
                analyser.select_ROIs()

        # Analyse movements if not analysed, othewise load these
        for analyser in analysers:
            if analyser.is_measured() == False or args.recalculate_movements:
                analyser.measure_movement(eye='left')
                analyser.measure_movement(eye='right')
            analyser.load_analysed_movements()
        
        
        if args.average:
            
            if len(analysers) >= 2:

                avg_analyser = MAverager(analysers)
                avg_analyser.setInterpolationSteps(5,5)
                
                analysers = avg_analyser
            else:
                analysers = analysers[0]
        else:
            if len(analysers) == 1:
                analysers = analysers[0]

        analyser_groups.append(analysers)
    

    function = analyses[args.analysis[0]]
    
    print(analyser_groups)
    
    kwargs = {}
    if wanted_imagefolders:
        kwargs['wanted_imagefolders'] = wanted_imagefolders
    if args.output:
        kwargs['save_fn'] = args.output
    if args.analysis_options:
        for aopts in args.analysis_options:
            for key, value in parse_key_value_pairs(aopts, splitchar=';;;').items():
                kwargs[key] = convert_kwarg(key, value)

    if function in MULTIANALYSER_CMDS.values():
        for analysers in analyser_groups:
            function(analysers, **kwargs)
    elif args.average or function in DUALANALYSER_CMDS.values():
        function(*analyser_groups, **kwargs)
    else:
        for analysers in analyser_groups:
            if not isinstance(analysers, list):
                analysers = [analysers]
            for analyser in analysers:
                function(analyser, **kwargs)

    if args.tk_waiting_window:
        waiting_window.close()
    
    if args.savefig:
        if args.output:
            plt.savefig(args.output, transparent=True)
        else:
            plt.savefig('figure.png', transparent=True)
    else:
        if not args.dont_show:
            plt.show()


if __name__ == "__main__":
    main()
