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

from .analysing import MAnalyser, MAverager
from .plotting import MPlotter, complete_flow_analysis
from .optic_flow import flow_direction, flow_vectors, field_error
from droso import DrosoSelect
from movie import Encoder
from image_adjusting import ROIAdjuster
from directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR_BIGFILES

from .new_analysing import optic_flow_error


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
        
        
        if os.path.isdir(self.argv[1]):
            # If data_folder given as the first argv
            directories = [self.argv[1]]
        else:

           
            if data_folder is None:
                selector = DrosoSelect()
                directories = selector.askUser(startswith='DrosoM')
            else:
                directories = data_folder
        
        analysers = []
            
        # Set up analysers at the selected DrosoM folders
        for directory in directories: 
            path, folder_name = os.path.split(directory)
            analyser = MAnalyser(path, folder_name) 
            analysers.append(analyser)
            

        # Ask ROIs if not selected
        for analyser in analysers:
            if not analyser.isROIsSelected():
                analyser.selectROIs()

        # Analyse movements if not analysed, othewise load these
        for analyser in analysers:

            if not analyser.isMovementsAnalysed() == (True, True) or 'recalculate' in self.argv:
                analyser.analyseMovement(eye='left')
                analyser.analyseMovement(eye='right')
            analyser.loadAnalysedMovements()
        
        
        plotter = MPlotter()

        # Plot results if asked so
        for analyser in analysers:
            if 'timeplot' in self.argv:
                analyser.timePlot()
            if 'magtrace' in self.argv:
                plotter.plotTimeCourses(analyser)
            if '2d_vectormap' in self.argv:
                plotter.plotDirection2D(analyser)
            
            if 'trajectories' in self.argv:
                plotter.plot_2d_trajectories(analyser)
            
            if 'vectormap' in self.argv:
                plotter.plot_3d_vectormap(analyser)

            if 'magnitude' in self.argv:
                plotter.plotMagnitude2D(analyser)

            if 'movie' in self.argv:
                print(analyser.getFolderName())
                images, ROIs = analyser.getTimeOrdered()
                
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

        if 'averaged' in self.argv:
            avg_analyser = MAverager(analysers)
            avg_analyser.setInterpolationSteps(5,5)
            #plotter.plotDirection2D(avg_analyser)
           
            if 'animation' in self.argv:
                animation = []
                step = 0.5
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
            else:
                animation = False

            if 'optimal_optic_flow' in self.argv:
                
                #points, measured_vecs = avg_analyser.get_3d_vectors('left')
                #measured_vecs = [np.array(v[1])-np.array(v[0]) for v in vectors_3d]
                #
                ##for point, vec in vectors_3d:
                ##    points.append( np.array([x[0], y[0], z[0]]) )
                ##    measured_vecs.append( np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]]) )


                #rotations = np.linspace(-180, 180, 100)
                #
                #errors = []
                #for rot in rotations: 
                #    flow_vecs = [flow_direction(P0, xrot=rot) for P0 in measured_vecs]
                #    
                #    
                #    # Errors for this rotation
                #    rot_errors = field_error(measured_vecs, flow_vecs)
                #    

                #    er = np.mean(rot_errors)
                #    print('Error of {} for rotation {}deg'.format(er, rot))
                #    errors.append(er)

                #plt.plot(rotations, errors)
                #plt.show()
                #
                #plotter.plot_3d_vectormap(avg_analyser,
                #        with_optic_flow=rotations[np.argmin(errors)], animation=animation)
                pass
            if 'complete_flow_analysis':
                
                rotations = np.linspace(-180,180, 360)

                complete_flow_analysis(avg_analyser, rotations)

            else:
                if 'mayavi' in self.argv:
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
        
        if 'show' in self.argv:
            plt.show()

        

def main():
    terminal = TerminalDrosoM()
    terminal.main()

if __name__ == "__main__":
    main()
