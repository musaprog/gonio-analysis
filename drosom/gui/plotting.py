import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.widgets
import tifffile

from tk_steroids.matplotlib import CanvasPlotter


class RecordingPlotter:
    '''
    Plotting single image folder data on the tk_steroids.matplotlib.CanvasPlotter.

    Tkinter intependend in sense that CanvasPlotter can easily be reimplemented on
    any other GUI toolkit (it's just a fusion of tkinter Canvas and matplotlib figure).
    
    -----------
    Attributes
    -----------
    i_repeat        If
    '''

    def __init__(self):
        
        self.analyser = None
        self.selected_recording = None

        self.colorbar = None
        self.roi_rectangles = []
        
        self.N_repeats = 0
        self.i_repeat = None

    def set_analyser(self, analyser):
        '''
        Set the current MAnalyser object.
        Returns None
        '''
        self.analyser = analyser


    def set_recording(self, selected_recording):
        '''
        Set the current selected_recording, ie. the name of the image folder (string).
        Returns None
        '''
        if self.selected_recording != selected_recording:
            self.i_repeat = None
        
        self.selected_recording = selected_recording
        if self.analyser.is_measured():
            self.movement_data = self.analyser.get_movements_from_folder(selected_recording)
            self.N_repeats = len(next(iter(self.movement_data.values())))
        else:
            self.movement_data = {}
            self.N_repeats = 0
    

    def magnitude(self, ax):
        '''
        Plot a displacement over time of the current specimen/recording.
        '''
        self.magnitudes = []
        
        for eye, movements in self.movement_data.items():
            for repetition in range(len(movements)):
                
                if (self.i_repeat is not None) and self.i_repeat != repetition:
                    continue

                mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                ax.plot(mag, label=str(repetition))
                
                self.magnitudes.append(mag)
        
        
        ax.legend(fontsize='xx-small', labelspacing=0.1, ncol=int(self.N_repeats/10)+1, loc='upper left')    
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Displacement sqrt(x^2+y^2) (pixels)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

 
    def xy(self, ax):
        '''
        Plot (x, y) where time is encoded by color.
        '''
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


    def ROI(self, ax):
        '''
        Plot specimen/recording image, and the ROIs and imaging parameters on top of it.
        '''
        self.roi_ax = ax
        fig = ax.get_figure()
        
        try:
            self.slider_ax
        except AttributeError:
            self.slider_ax = fig.add_axes([0.2, 0, 0.6, 0.1])

        image_fn = os.path.join(self.analyser.get_specimen_directory(), self.selected_recording, self.analyser.list_images(self.selected_recording)[0])
        self.image = tifffile.imread(image_fn)
        
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
        

        for roi in self.analyser.get_rois(self.selected_recording):
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
        
        imaging_params = self.analyser.get_imaging_parameters(self.selected_recording)
        if imaging_params:
            text = '\n'.join
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

