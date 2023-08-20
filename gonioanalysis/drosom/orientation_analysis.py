'''
Rhabdomere orientation
'''
import os
import json

import matplotlib.pyplot as plt

from roimarker import Marker

from gonioanalysis.drosom.analysing import MAnalyser
from gonioanalysis.directories import PROCESSING_TEMPDIR


class OAnalyser(MAnalyser):
    '''
    Rhabdomere orientation analyser.

    Inherits from MAnalyser though most of it's methods
    have no meaning.
    
    Measure movement opens Marker to draw lines/arrows

    '''

    def __init__(self, *args, **kwargs):


        super().__init__(*args, **kwargs)
        
        self._movements_skelefn = self._movements_skelefn.replace('movements_', 'orientation_')
        self.active_analysis = ''


    def measure_movement(self, eye, *args, **kwargs):
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
            
            if roi is not None:
                images.append(self.stacks[angle][0][0])
                

                extended_roi = [roi[0]-roi[2]/2, roi[1]-roi[3]/2, 2*roi[2], 2*roi[3]]

                rois.append(extended_roi)

        fig, ax = plt.subplots(num='Draw arrows for the {} eye'.format(eye))
        marker = Marker(fig, ax, images, self._movements_savefn.format(eye),
                relative_fns_from=os.path.join(self.data_path, self.folder),
                drop_imagefn=True,
                selection_type='arrow',
                crops=rois,
                callback_on_exit=lambda eye=eye: self._hotedit_marker_output(eye))

        marker.run()

        print('Marker should run now')


    def _hotedit_marker_output(self, eye):
        '''
        Edits Marker output to be Movemeter like output.
        '''

        with open(self._movements_savefn.format(eye), 'r') as fp:
            marker_data = json.load(fp)

        edited_data = {}

        for image_folder, arrows in marker_data.items():
            
            repeats = []

            for arrow in arrows:

                if len(arrow) == 2:
                    # Already edited, arrow is a dict we earlier made below
                    repeats.append( arrow )
                else:
                    # Needs hotediting
                    x1, y1, x2, y2 = arrow
                    repeats.append( {'x': [0, x1-x2], 'y': [0, y1-y2]} )

            # drop pos prefix [3:]
            if repeats != []:
                edited_data[image_folder[3:]] = repeats
       

        with open(self._movements_savefn.format(eye), 'w') as fp:
            json.dump(edited_data, fp)

   
    
    def is_measured(self):
        fns = [self._movements_savefn.format(eye) for eye in self.eyes]
        return all([os.path.exists(fn) for fn in fns])


