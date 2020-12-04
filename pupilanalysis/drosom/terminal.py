#!/usr/bin/env python3
'''
Analyse Pupil-Imsoft data and output the results.
'''
import sys
import os
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pupilanalysis.drosom import analyser_commands
from pupilanalysis.drosom.analyser_commands import ANALYSER_CMDS, DUALANALYSER_CMDS
from pupilanalysis.directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR_BIGFILES
from pupilanalysis.droso import DrosoSelect
from pupilanalysis.antenna_level import AntennaLevelFinder
from pupilanalysis.drosom.analysing import MAnalyser, MAverager
from pupilanalysis.drosom.orientation_analysis import OAnalyser
from pupilanalysis.drosom.optic_flow import FAnalyser
from pupilanalysis.drosom import plotting
from pupilanalysis.drosom.plotting.plotter import MPlotter
from pupilanalysis.drosom.plotting import complete_flow_analysis, error_at_flight
from pupilanalysis.drosom.special.norpa_rescues import norpa_rescue_manyrepeats
from pupilanalysis.drosom.special.paired import cli_group_and_compare
import pupilanalysis.drosom.reports as reports


if '--tk_waiting_window' in sys.argv:
    from pupilanalysis.tkgui.waiting_window import WaitingWindow



Analysers = {'orientation': OAnalyser, 'motion': MAnalyser, 'flow': FAnalyser}

analyses = {**ANALYSER_CMDS, **DUALANALYSER_CMDS}


def roimovement_video(analyser):
    '''
    Create a video where the imaging data is played and the analysed ROI is moved on the
    image, tracking the moving feature.

    Good for confirming visually that the movment analysis works.
    '''

    print(analyser.getFolderName())
    images, ROIs, angles = analyser.get_time_ordered()
    
    workdir = os.path.join(PROCESSING_TEMPDIR_BIGFILES, 'movie_{}'.format(str(datetime.datetime.now())))
    os.makedirs(workdir, exist_ok=True)

    newnames = [os.path.join(workdir, '{:>0}.jpg'.format(i)) for i in range(len(images))]
    

    adj = ROIAdjuster()
    newnames = adj.writeAdjusted(images, ROIs, newnames, extend_factor=3, binning=1)
    
    enc = Encoder()
    fps = 25
    enc.encode(newnames, os.path.join(ANALYSES_SAVEDIR, 'movies','{}_{}fps.mp4'.format(analyser.getFolderName(), fps)), fps)
 
    for image in newnames:
        os.remove(image)
    try:
        os.rmdir(workdir)
    except OSError:
        print("Temporal directory {} left behind because it's not empty".format(workdir))
       

   
def export_optic_flow():
    '''
    Exports the optic flow vectors.
    '''
    import json
    from pupilanalysis.coordinates import optimal_sampling
    from pupilanalysis.drosom.optic_flow import flow_vectors
    points = optimal_sampling(np.arange(-90, 90, 5), np.arange(-180, 180, 5))
    vectors = flow_vectors(points)
    
    with open('optic_flow_vectors.json', 'w') as fp:
        json.dump({'points': np.array(points).tolist(), 'vectors': np.array(vectors).tolist()}, fp)
            


         
def main(custom_args=None):
    
    if custom_args is None:
        custom_args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    
    # DATA ARGUMENTS
    parser.add_argument('-D', '--data_directory', nargs=1,
            help='Data directory')

    parser.add_argument('-S', '--specimens', nargs='+',
            help='Comma separeted list of specimen names. Separate groups by space when averaging is on.')

    

    # Analyser settings
    parser.add_argument('-a', '--average', action='store_true',
            help='Average and interpolate the results over the specimens')
    
    parser.add_argument('--short-name', nargs=1,
            help='Short name to set if --average is set')

    parser.add_argument('-t', '--type', nargs='+',
            help='Analyser type, either "motion" or "orientation". Space separate gor groups')
    
    parser.add_argument('-r', '--reselect-rois', action='store_true',
            help='Reselect ROIs')
 
    parser.add_argument('-R', '--recalculate-movements', action='store_true',
            help='Recalculate with Movemeter')

    parser.add_argument('--reverse-directions', action='store_true',
            help='Reverse movement directions')
    
    # Other settings
    parser.add_argument('--tk_waiting_window', action='store_true',
            help='(internal) Launches a tkinter waiting window')
    parser.add_argument('--dont-show', action='store_true',
            help='Skips showing the plots')
    parser.add_argument('--worker-info', nargs=2,
            help='Worker id and total number of parallel workers. Only 3D video plotting now')

    # Different analyses for separate specimens

    parser.add_argument('-A', '--analysis', nargs=1,
            choices=analyses.keys(),
            help='Analysis method or action. Allowed analyses are '+', '.join(analyses.keys()))

    args = parser.parse_args(custom_args)
    



    if args.tk_waiting_window:
        waiting_window = WaitingWindow('terminal.py', 'When ready, this window closes.')

    if args.worker_info:
        analyser_commands.I_WORKER = int(args.worker_info[0])
        analyser_commands.N_WORKERS = int(args.worker_info[1])

    # Getting the data directory
    # --------------------------
    if args.data_directory:
        print('Using data directory {}'.format(args.data_directory[0]))
        
        data_directory = args.data_directory[0]
    else:
        data_directory = input('Data directory >> ')
    
    # Check that the data directory exists
    if not os.path.isdir(data_directory):
        raise ValueError("{} is not a directory".format(data_directory))


    # Getting the specimens
    # ---------------------
    directory_groups = []
    if args.specimens:
        
        for group in args.specimens:
            print('Using specimens {}'.format(group))
            
            if group == 'none':
                directory_groups.append(None)
                continue

            selector = DrosoSelect(datadir=data_directory)
            directories = selector.parse_specimens(group)
            directory_groups.append(directories)
    else:
        selector = DrosoSelect(datadir=data_directory)
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
                analysers.append(analyser)

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
        
        
        if args.reverse_directions:
            for analyser in analysers:
                analyser.receptive_fields = True
        
        

        if args.average:
            
            if len(analysers) >= 2:

                avg_analyser = MAverager(analysers)
                avg_analyser.setInterpolationSteps(5,5)
                
                if args.short_name:
                    avg_analyser.set_short_name(args.short_name[0])
                           
                analysers = avg_analyser
            else:
                analysers = analysers[0]

        analyser_groups.append(analysers)
    

    function = analyses[args.analysis[0]]
    
    print(analyser_groups)

    if args.average:
        function(*analyser_groups)
    else:
        for analysers in analyser_groups:
            for analyser in analysers:
                function(analyser)

    if args.tk_waiting_window:
        waiting_window.close()

    if not args.dont_show:
        plt.show()


if __name__ == "__main__":
    main()
