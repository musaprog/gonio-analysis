
import os
import json

import numpy as np
import tifffile

from gonioanalysis.drosom.analysing import MAnalyser



class TAnalyser(MAnalyser):
    '''
    Transmittance analyser.

    Analyses the ROI's light throughput while the ROI tracks
    its moving feature (ie. using MAnalyser motion analysis results)
    
    Mean (average) pixel value of the ROI is quantified.
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self._real_movements_savefn = self._movements_savefn

        self._movements_skelefn = self._movements_skelefn.replace('movements_', 'transmittance_')
        self.active_analysis = ''


    def measure_movement(self, eye, *args, **kwargs):
        '''
        Analyse transmittance/brightness by calculating mean (average)
        pixel value of the ROI in its time locations, and save results.
        '''
        self.movements = {}
        
        intensities = {}

        manalyser = MAnalyser(self.data_path, self.folder)
        manalyser.active_analysis = self.active_analysis

        for i, angle in enumerate(self.stacks):
            print('  Image folder {}/{}'.format(i+1, len(self.stacks)))
            
            roi = self.ROIs[eye].get(angle, None)

            if roi is not None:
                
                images = self.stacks[angle]
                
                intensities[angle] = []

                for i_repeat, repeat in enumerate(images):
                    ints = []
                    
                    try:
                        _roi = manalyser.get_moving_ROIs(eye, angle, i_repeat)
                    except AttributeError:
                        _roi = None
                    
                    i_frame = 0

                    for fn in repeat:
                        
                        tiff = tifffile.TiffFile(fn)
                        
                        for i_page in range(len(tiff.pages)):
                            
                            images = tiff.asarray(key=i_page)
                            if len(images.shape) == 2:
                                images = [images]
                            
                            for image in images:
                                if _roi is None:
                                    x,y,w,h = roi
                                else:
                                    try:
                                        x,y,w,h = [int(round(z)) for z in _roi[i_frame]]
                                    except IndexError:
                                        # No ROI movement, skip
                                        break
                                
                                intensity = np.mean(image[y:y+h,x:x+w])
                                ints.append(intensity)
                                
                                i_frame += 1
                                print("fn {}: {}/{}".format(os.path.basename(fn), i_frame+1, len(tiff.pages)))

                    # X component carries the brightness change information
                    # Y component is set to zero for compability (in future
                    #   remove to save disk?)
                    intensities[angle].append({'x': ints, 'y':[0]*len(ints)})


        self.movements[eye] = intensities

        # Save movements
        with open(self._movements_savefn.format(eye), 'w') as fp:
            json.dump(intensities, fp)

    

    def is_measured(self):
        fns = [self._movements_savefn.format(eye) for eye in self.eyes]
        return all([os.path.exists(fn) for fn in fns])


    def load_analysed_movements(self, *args, **kwargs):
        super().load_analysed_movements(*args, **kwargs)

        # Nullify the Y component so that X carries the brightness change data.
        # Otherwise, at get_magnitude traces, when calculating displacement
        # distance, the data gets multiplied by sqrt(2).
        #
        # The same change is done in TAnalyser.measure_movement but this also
        # fixes previously analysed data without reanalysing.
        for eye in self.movements:
            for image_folder, data in self.movements[eye].items():
                for i_repeat in range(len(data)):
                    data[i_repeat]['y'] = np.zeros(len(data[i_repeat]['x']))


    # Here the original MAnalyser method modified so that the presentation
    # makes more sense with light transmittance analysis 
    #
    # 1) The first data point is often "polluted" because the stimulus LED
    #   goes on within the first frame of imaging (thus first frame darker)
    #   -> bring the 1st to the mean of 2nd and 3rd
    # 2) Shift the trace in Y to always start from zero
    #
    def get_magnitude_traces(self, *args, **kwargs):
        
        magnitude_traces = super().get_magnitude_traces(*args, **kwargs)

        for image_folder in magnitude_traces:

            # 1) "Recompute" the first data point
            for trace in magnitude_traces[image_folder]:
                trace[0] = np.mean((trace[1], trace[2]))

            # 2) Shift whole trace
            magnitude_traces[image_folder] = np.array([t-t[0] for t in magnitude_traces[image_folder]])

        return magnitude_traces

