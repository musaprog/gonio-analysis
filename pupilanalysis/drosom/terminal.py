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

from pupilanalysis.drosom.analyser_commands import ANALYSER_CMDS as analyses
from pupilanalysis.directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR_BIGFILES
from pupilanalysis.droso import DrosoSelect
from pupilanalysis.antenna_level import AntennaLevelFinder
from pupilanalysis.drosom.analysing import MAnalyser, MAverager
from pupilanalysis.drosom.orientation_analysis import OAnalyser
from pupilanalysis.drosom import plotting
from pupilanalysis.drosom.plotting.plotter import MPlotter
from pupilanalysis.drosom.plotting import complete_flow_analysis, error_at_flight
from pupilanalysis.drosom.special.norpa_rescues import norpa_rescue_manyrepeats
from pupilanalysis.drosom.special.paired import cli_group_and_compare
import pupilanalysis.drosom.reports as reports


if 'tk_waiting_window' in sys.argv:
    from .gui.waiting_window import WaitingWindow



Analysers = {'orientation': OAnalyser, 'motion': MAnalyser}



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
    from pupilanalysis.optimal_sampling import optimal as optimal_sampling
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

    parser.add_argument('-S', '--specimens', nargs=1,
            help='Comma separeted list of specimen names')

    

    # Analyser settings
    parser.add_argument('-a', '--average', action='store_true',
            help='Average and interpolate the results over the specimens')

    parser.add_argument('-t', '--type', default='motion', choices=['motion', 'orientation'],
            help='Analyser type, either "motion" or "orientation"')
    
    parser.add_argument('-r', '--reselect-rois', action='store_true',
            help='Reselect ROIs')
 
    parser.add_argument('-R', '--recalculate-movements', action='store_true',
            help='Recalculate with Movemeter')

    parser.add_argument('--reverse-directions', action='store_true',
            help='Reverse movement directions')
    
    # Other settings
    parser.add_argument('--tk_waiting_window', action='store_true',
            help='(internal) Launches a tkinter waiting window')
    parser.add_argument('--unittest', action='store_true',
            help='Skips showing the plots')

    # Different analyses for separate specimens

    parser.add_argument('analysis', metavar='analysis',
            choices=analyses.keys(),
            help='Analysis method or action. Allowed analyses are '+', '.join(analyses.keys()))

    args = parser.parse_args(custom_args)
    



    if args.tk_waiting_window:
        self.waiting_window = WaitingWindow('terminal.py', 'When ready, this window closes.')


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
    if args.specimens:
        print('Using specimens {}'.format(args.specimens))

        selector = DrosoSelect(datadir=data_directory)
        directories = selector.parse_specimens(args.specimens[0])
    else:
        selector = DrosoSelect(datadir=data_directory)
        directories = selector.ask_user()
     
            
    # Setting up analysers
    # ---------------------
    analysers = []
    Analyser = Analysers[args.type]

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
        avg_analyser = MAverager(analysers)
        avg_analyser.setInterpolationSteps(5,5)
        
        short_name = [arg.split('=')[1] for arg in self.argv if 'short_name=' in arg]
        if short_name:
            avg_analyser.set_short_name(short_name[0])
       
        analysers = [avg_analyser]
    
    
    function = analyses[args.analysis]
    
    for analyser in analysers:
        function(analyser)

    if args.tk_waiting_window:
        self.waiting_window.close()

    if not args.unittest:
        plt.show()


if __name__ == "__main__":
    main()
