import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.widgets
import tifffile

from tk_steroids.matplotlib import CanvasPlotter

from gonioanalysis.drosom.plotting.basics import (
        plot_1d_magnitude,
        plot_xy_trajectory,
        plot_3d_vectormap
        )

class RecordingPlotter:
    '''
    Plotting single image folder data on the tk_steroids.matplotlib.CanvasPlotter.

    Tkinter intependend in sense that CanvasPlotter can easily be reimplemented on
    any other GUI toolkit (it's just a fusion of tkinter Canvas and matplotlib figure).
    
    -----------
    Attributes
    -----------
    self.core
        Reference to the Core instance
    self.selected_recording
        The recording currently plotted at the plotter
    self.i_repeat
        None if show all traces, otherwise the index of repeat to be shown
    self.N_repeats
        The total number of repeats in the image folder.

    '''

    def __init__(self, core):
        '''
        core    An instance of the Core class in core.py
        '''

        self.core = core
        
        # Keeps internally track of the current recording on plot
        self.selected_recording = None

        self.colorbar = None
        self.roi_rectangles = []
        
        self.N_repeats = 0
        self.i_repeat = None


    def _check_recording(self, skip_datafetch=False):
        '''
        Check from core if the selected has changed.
        '''
        selected_recording = self.core.selected_recording

        if self.selected_recording != selected_recording:
            self.i_repeat = None
        
        self.selected_recording = selected_recording

        if not skip_datafetch:
            if self.core.analyser.is_measured():
                self.movement_data = self.core.analyser.get_movements_from_folder(selected_recording)
                self.N_repeats = len(next(iter(self.movement_data.values())))
                pass
            else:
                self.movement_data = {}
                self.N_repeats = 0
        


    def magnitude(self, ax, **kwargs):
        '''
        Plot a displacement over time of the current specimen/recording.
        '''
        self._check_recording(skip_datafetch=True)
        
        ax, self.magnitudes, self.N_repeats = plot_1d_magnitude(self.core.analyser,
                self.selected_recording,
                i_repeat=self.i_repeat,
                label='EYE-repIREPEAT',
                ax=ax,
                **kwargs)


    def vectormap(self, ax, **kwargs):

        self.N_repeats = 0
        
        ax, self.vectors = plot_3d_vectormap(self.core.analyser,
                ax=ax,
                **kwargs)



    def xy(self, ax, **kwargs):
        '''
        Plot (x, y) where time is encoded by color.
        '''
        self._check_recording()
        
        ax, self.xys = plot_xy_trajectory([self.core.analyser],
                {self.core.analyser.name: [self.selected_recording]},
                i_repeat=self.i_repeat,
                ax=ax,
                **kwargs)

    
    def ROI(self, ax):
        '''
        Plot specimen/recording image, and the ROIs and imaging parameters on top of it.
        '''
        self._check_recording(skip_datafetch=True)
        
        self.roi_ax = ax
        fig = ax.get_figure()
        
        try:
            self.slider_ax
        except AttributeError:
            self.slider_ax = fig.add_axes([0.2, 0.01, 0.6, 0.05])
        
        # Get a list of image filenames and how many
        image_fns = self.core.analyser.list_images(self.selected_recording, absolute_path=True)
        self.N_repeats = len(image_fns)

        if self.i_repeat:
            i_frame = self.i_repeat
        else:
            i_frame = 0
        
        image_fn = image_fns[i_frame]
        self.image = tifffile.TiffFile(image_fn).asarray(key=0)

        try:
            self.range_slider
        except AttributeError:
            self.range_slider = matplotlib.widgets.Slider(self.slider_ax, 'Range %' , 0, 100, valinit=90, valstep=1)
            self.range_slider.on_changed(self.update_ROI_plot)
        
        # Draw ROI rectangles
        for old_roi in self.roi_rectangles:
            try:
                old_roi.remove()
            except NotImplementedError:
                # This error occurs when something goes from from here on before
                # the ax.add_patch(roi) line. We can just ignore this as the
                # ROI has not been ever added to the ax
                continue
        self.roi_rectangles = []
        

        for roi in self.core.analyser.get_rois(self.selected_recording):
            patch = matplotlib.patches.Rectangle((roi[0], roi[1]), roi[2], roi[3],
                    fill=False, edgecolor='White')
            self.roi_rectangles.append(patch)
        
        self.update_ROI_plot(self.range_slider.val)

        for roi in self.roi_rectangles:
            ax.add_patch(roi)


    def update_ROI_plot(self, slider_value):
        '''
        This gets called when the brightness cap slider is moved.
        '''
        clipped = np.clip(self.image, 0, np.percentile(self.image, slider_value))
        clipped /= np.max(clipped)

        try:
            self.roi_imshow.set_data(clipped)
        except AttributeError:
            self.roi_imshow = self.roi_ax.imshow(clipped, cmap='gray')
        
        imaging_params = self.core.analyser.get_imaging_parameters(self.selected_recording)
        if imaging_params:
            text = '\n'.join(['{}: {}'.format(setting, value) for setting, value in imaging_params.items()])
        else:
            text = 'Unable to fetch imaging parameters'

        try:
            self.roi_text.set_text(text)
        except AttributeError:
            self.roi_text = self.roi_ax.text(0,1, text, ha='left', va='top', fontsize='small', 
                    transform=self.roi_ax.transAxes)



def main():
    pass


if __name__ == "__main__":
    main()

