'''
Rhabdomere orientation
'''
import os
import json

import matplotlib.pyplot as plt

from roimarker import Marker

from pupilanalysis.drosom.analysing import MAnalyser
from pupilanalysis.directories import PROCESSING_TEMPDIR


class OAnalyser(MAnalyser):
    '''
    Rhabdomere orientation analyser.

    Inherits from MAnalyser though most of it's methods
    have no meaning.
    
    Measure movement opens Marker to draw lines/arrows

    '''

    def __init__(self, *args, **kwargs):

        self.orientation_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', args[1], 'orientation_{}_{}.json'.format(args[1], '{}'))
        os.makedirs(os.path.dirname(self.orientation_savefn), exist_ok=True)

        print("Orientation saved in file {}".format(self.orientation_savefn))

        super().__init__(*args, **kwargs)


    def measure_movement(self, eye):
        '''
        The measure movement method overridden to meausure the (rhabdomere)
        orientation.

        In the end calls self.load_analysed_movements in order to
        match the MAnalyser behaviour.
        '''

        self.movements = {}

        images = []
        rois = []
            
        for angle in self.stacks:
            
            roi = self.ROIs[eye].get(angle, None)
            
            if roi:
                images.append(self.stacks[angle][0][0])
                rois.append(roi)


        fig, ax = plt.subplots()
        marker = Marker(fig, ax, images[1:10], self.orientation_savefn.format(eye),
                relative_fns_from=os.path.join(self.data_path, self.folder),
                selection_type='arrow',
                callback_on_exit=print)

        marker.run()

        print('Marker should run now')

    def load_analysed_movements(self):
        '''
        '''


        self.movements = {}

        for eye in ['left', 'right']:
            
            self.movements[eye] = {}

            with open(self.orientation_savefn.format(eye), 'r') as fp:
                marker_data = json.load(fp)
            
            for angle in self.stacks:
                
                roi = marker_data.get( self.stacks[angle][0][0] )
                
                try:
                    self.movements[angle]
                except KeyError:
                    self.movements[angle] = []
                
                x = roi[2] - roi[0]
                y = roi[3] - roi[1]

                self.movements[eye][angle].append({'x': x, 'y': y})

    
    def is_measured(self):
        return os.path.exists(self.orientation_savefn) 


