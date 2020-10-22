#!/usr/bin/env python3
'''
Terminal program to use DrosoM tools.
'''
import sys
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

# Plotting 3D in matplotlib
from mpl_toolkits.mplot3d import Axes3D

from pupilanalysis.directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR_BIGFILES
from pupilanalysis.droso import DrosoSelect
from pupilanalysis.antenna_level import AntennaLevelFinder
from pupilanalysis.drosom.analysing import MAnalyser, MAverager
from pupilanalysis.drosom.orientation_analysis import OAnalyser
from pupilanalysis.drosom.plotting import MPlotter, complete_flow_analysis, error_at_flight
import pupilanalysis.drosom.plotting as plotting
from pupilanalysis.drosom.optic_flow import flow_direction, flow_vectors, field_error
from pupilanalysis.drosom.special.norpa_rescues import norpa_rescue_manyrepeats
from pupilanalysis.drosom.special.paired import cli_group_and_compare
import pupilanalysis.drosom.reports as reports

#from videowrapper import Encoder


if 'tk_waiting_window' in sys.argv:
    from .gui.waiting_window import WaitingWindow


def make_animation_angles():
    '''
    Returns the matplotlib angles to rotate a 3D plot
    
    This really shouldnt be here...
    '''

    animation = []
    step = 0.5 # old 0.5
    sidego = 30
    # go up, to dorsal
    for i in np.arange(-30,60,step):
        animation.append((i,90))
    #rotate azim
    for i in np.arange(90,90+sidego,step*2):
        animation.append((60,i))
    # go back super down, to ventral
    for i in np.arange(0,120,step):
        animation.append((60-i,90+sidego))
    # rotate -azim
    for i in np.arange(0,2*sidego,step*2): 
        animation.append((-60,90+sidego-i))
    # go up back to dorsal
    for i in np.arange(0,120, step):
        animation.append((-60+i,90-sidego))
    return animation


class TerminalDrosoM:
    '''
    Using drosom.py from terminal (command line).
    '''


    def __init__(self, custom_args=None): 
        
        # These determine what is considered proper 
        if custom_args is None:
            self.argv = sys.argv
        else:
            self.argv = custom_args


    def help(self):
        '''
        Prints the help when the script is ran from terminal.
        '''

        print('The following arguments are supported:')
        arguments = ['3dplot Creates an interactive 3D', 'averaged']
    
        for line in arguments:
            print('  {}'.format(line))

    
    def main(self, data_folder=None):
        
        if 'tk_waiting_window' in self.argv:
            self.waiting_window = WaitingWindow('terminal.py', 'When ready, this window closes.')

        if len(self.argv)>1 and os.path.isdir(self.argv[1]):
            # If data_folder given as the firsts argvs
            directories = [arg for arg in self.argv[1:] if os.path.isdir(arg)]
        else:

           
            if data_folder is None:
                selector = DrosoSelect()
                directories = selector.ask_user()
            else:
                directories = data_folder
        
        analysers = []
            
        # Set up analysers at the selected DrosoM folders
        for directory in directories: 
            
            if 'antenna_level' in self.argv:
                AntennaLevelFinder().find_level(directory)
              
            path, folder_name = os.path.split(directory)
            
            # Use either MAnalyser (default), or OAnalyser for
            # rhabdomere orientation analysis.
            if 'orientation_analysis' in self.argv:
                analyser = OAnalyser(path, folder_name)
            else:
                analyser = MAnalyser(path, folder_name)

            analysers.append(analyser)
            

        # Ask ROIs if not selected
        for analyser in analysers:
            if not analyser.are_rois_selected():
                analyser.select_ROIs()

        # Analyse movements if not analysed, othewise load these
        for analyser in analysers:
            if analyser.is_measured() == False or 'recalculate' in self.argv:
                analyser.measure_movement(eye='left')
                analyser.measure_movement(eye='right')
            analyser.load_analysed_movements()
    
        if 'receptive_fields' in self.argv:
            for analyser in analysers:
                analyser.receptive_fields = True

   
        
        if 'animation' in self.argv:
            animation = make_animation_angles()
        else:
            animation = False
        

        plotter = MPlotter()

        # Plot results if asked so
        if not 'averaged' in self.argv:
            for analyser in analysers:
                if 'timeplot' in self.argv:
                    analyser.time_plot()
                if 'magtrace' in self.argv:
                    plotter.plotTimeCourses(analyser)
                
                if '2d_vectormap' in self.argv:
                    plotter.plotDirection2D(analyser)
                
                

                if 'trajectories' in self.argv:
                    plotter.plot_2d_trajectories(analyser)
                
                if 'vectormap' in self.argv:
                    plotter.plot_3d_vectormap(analyser, animation=animation)
                
                if '1dmagnitude' in self.argv:
                    folder = self.argv[-1]
                    plotter.plot_1d_magnitude_from_folder(analyser, folder)
                

                if '2dmagnitude' in self.argv:
                    plotter.plotMagnitude2D(analyser)

                if 'movie' in self.argv:
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

                if 'illustrate_experiments' in self.argv:
                    plotting.illustrate_experiments(analyser)

            if 'norpa_rescue_manyrepeats' in self.argv:
                norpa_rescue_manyrepeats(analysers)
            if 'paired_compare' in self.argv:
                cli_group_and_compare(analysers)


            if 'left_right_summary' in self.argv:
                reports.left_right_summary(analysers)

            if 'pdf_summary' in self.argv:
                reports.pdf_summary(analysers)


        if 'averaged' in self.argv:
            avg_analyser = MAverager(analysers)
            avg_analyser.setInterpolationSteps(5,5)
            
            short_name = [arg.split('=')[1] for arg in self.argv if 'short_name=' in arg]
            if short_name:
                avg_analyser.set_short_name(short_name[0])
            
            
           
            if 'export_vectormap' in self.argv:
                print('Exporting 3d vectors')
                avg_analyser.export_3d_vectors()

            elif 'export_optic_flow' in self.argv:
                #print('Exporting optic flow vectors')
                #avg_analyser.export_3d_vectors(optic_flow=True)

                import json
                from pupilanalysis.optimal_sampling import optimal as optimal_sampling
                from pupilanalysis.drosom.optic_flow import flow_vectors
                points = optimal_sampling(np.arange(-90, 90, 5), np.arange(-180, 180, 5))
                vectors = flow_vectors(points)
                
                with open('optic_flow_vectors.json', 'w') as fp:
                    json.dump({'points': np.array(points).tolist(), 'vectors': np.array(vectors).tolist()}, fp)


            elif 'magtrace' in self.argv:
                plotter.plotTimeCourses(avg_analyser)
            
            elif 'complete_flow_analysis' in self.argv:
                
                rotations = np.linspace(-180,180, 360)
                
                if 'pitch' in self.argv:
                    axis = 'pitch'
                elif 'yaw' in self.argv:
                    axis = 'yaw'
                elif 'roll' in self.argv:
                    axis = 'roll'
                
                complete_flow_analysis(avg_analyser, rotations, axis)

            elif 'error_at_flight' in self.argv:
                error_at_flight(avg_analyser)

            elif 'mayavi' in self.argv:
                plotter.plot_3d_vectormap_mayavi(avg_analyser)
            else:
                plotter.plot_3d_vectormap(avg_analyser, animation=animation)
            
            


        if 'averaged-magnitude' in self.argv:
            avg_analyser = MAverager(analysers)
            avg_analyser.setInterpolationSteps(10,10)
            plotter.plotMagnitude2D(avg_analyser)


        plotter.setLimits('common')

        if 'save' in self.argv:
            plotter.save()
        
         
        if 'tk_waiting_window' in self.argv:
            self.waiting_window.close()
        
        if 'animation' in self.argv:
            pass
        else:
            plt.savefig('fig.png', dpi=600)
            plt.show()

        

def main():
    terminal = TerminalDrosoM()
    terminal.main()

if __name__ == "__main__":
    main()
