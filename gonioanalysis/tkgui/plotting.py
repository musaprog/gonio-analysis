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



    def xy(self, ax):
        '''
        Plot (x, y) where time is encoded by color.
        '''
        self._check_recording()
        
        self.xys = []

        for eye, movements in self.movement_data.items():
            for repetition in range(len(movements)):
                
                if (self.i_repeat is not None) and self.i_repeat != repetition:
                    continue

               
                x = movements[repetition]['x']
                y = movements[repetition]['y']
                 
                N = len(movements[repetition]['x'])
                
                cmap = matplotlib.cm.get_cmap('inferno', N)
               
                for i_point in range(1, N):
                    ax.plot([x[i_point-1], x[i_point]], [y[i_point-1], y[i_point]], color=cmap((i_point-1)/(N-1)))
            
                ax.scatter(x[0], y[0], color='black')
                ax.scatter(x[-1], y[-1], color='gray')
                
            
                self.xys.append([x, y])


        # Colormap
        if self.movement_data:
            if not self.colorbar: 
                time = [i for i in range(N)]
                sm = matplotlib.cm.ScalarMappable(cmap=cmap)
                sm.set_array(time)


                fig = ax.get_figure()
                self.colorbar = fig.colorbar(sm, ticks=time, boundaries=time, ax=ax, orientation='horizontal')
                self.colorbar.set_label('Frame')
            else:
                #self.colorbar.set_clim(0, N-1)
                pass

        ax.set_xlabel('Displacement in X (pixels)')
        ax.set_ylabel('Displacement in Y (pixels)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_aspect('equal', adjustable='box')



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
        image_fns = self.core.analyser.list_images(self.selected_recording)
        self.N_repeats = len(image_fns)

        if self.i_repeat:
            i_frame = self.i_repeat
        else:
            i_frame = 0
        
        image_fn = os.path.join(self.core.analyser.get_specimen_directory(), self.selected_recording, image_fns[i_frame])
        self.image = tifffile.imread(image_fn)
        
        # Fast fix for opening stacks
        if len(self.image.shape) == 3:
            self.image = self.image[0,:,:]

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

