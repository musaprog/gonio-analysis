'''

TODO
+++ export all traces to clipboard
++ antenna_level finder
+ after movement measure, update MAnalyser
+ no need for manual add to PythonPath on Windows
- show statistics
- intensity series plotting
- reading imaging parameters for each imagefolder from the descriptions file
- multiselect ROIs (if many separate recordings at the same site)


'''
import os
import multiprocessing
import subprocess
import sys
import ctypes

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

# PLotting
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.widgets
import tifffile

from tk_steroids.elements import Listbox, Tabs, ButtonsFrame, TickSelect
from tk_steroids.matplotlib import CanvasPlotter

from pupil.directories import PROCESSING_TEMPDIR, PROCESSING_TEMPDIR_BIGFILES
from pupil.antenna_level import AntennaLevelFinder
from pupil.drosom.loading import angles_from_fn
from pupil.drosom.analysing import MAnalyser
from pupil.drosom.plotting import MPlotter
from pupil.drosom.gui.run_measurement import MeasurementWindow
from pupil.drosom.gui.core import Core

from pupil_imsoft.anglepairs import toDegrees




class ExamineMenubar(tk.Frame):
    '''
    Menubar class for the examine GUI.
    '''

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.root = parent.root
        self.core = parent.core
        self.menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Set data directory', command=self.parent.set_data_directory)
        file_menu.add_command(label='Exit', command=self.on_exit)
        self.menubar.add_cascade(label='File', menu=file_menu)

        # Batch run
        #batch_menu = tk.Menu(self.menubar, tearoff=0)
        #batch_menu.add_command(label='Select ALL ROIs', command=self.batch_ROIs)
        #batch_menu.add_command(label='Measure ALL movements', command=self.batch_measurement)
        #self.menubar.add_cascade(label='Batch', menu=batch_menu)
        
        
        # Data plotting
        plot_menu = tk.Menu(self.menubar, tearoff=0)
        plot_menu.add_command(label='Vectormap', command=lambda: self.core.adm_subprocess('current', 'vectormap'))
        plot_menu.add_command(label='Vectormap - rotating video', command=lambda: self.core.adm_subprocess('current', 'vectormap animation')) 
        plot_menu.add_separator()
        plot_menu.add_command(label='Displacement over time', command=lambda: self.core.adm_subprocess('current', 'magtrace'))
        self.menubar.add_cascade(label='Specimen', menu=plot_menu)
        self.plot_menu = plot_menu        
        
        # Data plotting
        many_menu = tk.Menu(self.menubar, tearoff=0)
        many_menu.add_command(label='Measure movements...', 
                command=lambda: self.select_specimens(self.batch_measure, with_rois=True))
        
         
        # Requiers adding get_magnitude_traces to MAverager
        
        #many_menu.add_command(label='Displacement over time', command=lambda: self.select_specimens(lambda specimens: self.core.adm_subprocess(specimens, 'averaged magtrace')))
        
        #many_menu.add_command(label='Select  movements...', command=lambda: )
        #
        
        many_menu.add_separator()
        many_menu.add_command(label='Averaged vectormap...', command=lambda: self.select_specimens(lambda specimens: self.core.adm_subprocess(specimens, 'averaged')))
        
        self.menubar.add_cascade(label='Many specimens', menu=many_menu)
        self.many_menu = many_menu        


        self.winfo_toplevel().config(menu=self.menubar)

    
    def batch_measure(self, specimens):
        targets = [self.core.get_manalyser(specimen).measure_both_eyes for specimen in specimens]
        MeasurementWindow(self.root, targets, title='Measure movement')
    
    def select_specimens(self, command, with_rois=False, with_movements=False):
        '''
        Opens specimen selection window and after ok runs command using
        selected specimens list as the only input argument.

        command         Command to close after fly selection
        with_rois       List specimens with ROIs selected if True
        with_movements  List specimens with movements measured if True
        '''
        top = tk.Toplevel()
        top.title('Select specimens')
        
        specimens = self.core.list_specimens(with_rois=with_rois, with_movements=with_movements) 
        selector = TickSelect(top, specimens, command)

        #        lambda specimens: self.core.adm_subprocess(specimens, 'averaged'))
        selector.grid()
        
        tk.Button(selector, text='Close', command=top.destroy).grid(row=1, column=1)


    def on_exit(self):
        self.winfo_toplevel().destroy()


    def update_states(self, manalyser):
        '''
        Updates menu entry states (enabled/disabled) based specimen's status (ROIs set, etc)
        '''
        
        rois = manalyser.is_rois_selected()
        measurements = manalyser.is_measured()
        
        # Plot menu
        if measurements and rois:
            state = tk.NORMAL
        else:
            state = tk.DISABLED
        self.plot_menu.entryconfig(0, state=state)        
        self.plot_menu.entryconfig(1, state=state)        
        


class ExamineView(tk.Frame):
    '''
    The examine frame. Selection of
    - data directory
    - specimen
    - recording
    and plotting the intemediate result for each recording.
    '''
    
    def __init__(self, parent):
        
        tk.Frame.__init__(self, parent)
        
        self.core = Core()
        
        self.root = self.winfo_toplevel()

        # Make canvas plotter to stretch
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


        #tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        # Uncomment to have the menubar
        self.menu = ExamineMenubar(self)

        # LEFTSIDE frame
        self.leftside_frame = tk.Frame(self)
        self.leftside_frame.grid(row=0, column=0, sticky='NS') 
        self.leftside_frame.grid_rowconfigure(4, weight=1) 

        # The 1st buttons frame, selecting root data directory
        self.buttons_frame_1 = ButtonsFrame(self.leftside_frame, ['Set data directory'],
                [self.set_data_directory, self.set_data_directory])
        self.buttons_frame_1.grid(row=0, column=0, sticky='NW', columnspan=2)

 
        self.specimen_control_frame = tk.LabelFrame(self.leftside_frame, text='Specimen')
        self.specimen_control_frame.grid(row=1, column=0, sticky='NWES', columnspan=2)

       
        # The 2nd buttons frame, ROIs and movements
        self.buttons_frame_2 = ButtonsFrame(self.specimen_control_frame,
                ['Select ROIs', 'Measure movement', 'Zero correct', 'Copy to clipboard'],
                [self.select_rois, self.measure_movement, self.antenna_level, self.copy_mean])
        self.buttons_frame_2.grid(row=1, column=0, sticky='NW', columnspan=2)
        self.button_rois, self.button_measure, self.button_zero, self.copy_mean = self.buttons_frame_2.get_buttons()
        self.copy_mean.grid(row=1, column=0, columnspan=3, sticky='W')
        # Subframe for 2nd buttons frame
        #self.status_frame = tk.Frame(self.leftside_frame)
        #self.status_frame.grid(row=2)

        self.status_rois = tk.Label(self.specimen_control_frame, text='ROIs selected 0/0', font=('system', 8))
        self.status_rois.grid(row=2, column=0, sticky='W')
        
        self.status_antenna_level = tk.Label(self.specimen_control_frame, text='Zero correcter N/A', font=('system', 8))
        self.status_antenna_level.grid(row=3, column=0, sticky='W')
        
        
        
        # Image folder manipulations
        self.folder_control_frame = tk.LabelFrame(self.leftside_frame, text='Image folder')
        self.folder_control_frame.grid(row=2, column=0, sticky='NWES', columnspan=2)
       
        self.buttons_frame_3 = ButtonsFrame(self.folder_control_frame,
                ['Reselect ROI', 'Copy to clipboard'],
                [self.select_roi, self.copy_to_clipboard])
        self.buttons_frame_3.grid(row=1, column=0, sticky='NW', columnspan=2)
        self.button_one_roi = self.buttons_frame_2.get_buttons()[0]
        

        self.status_horizontal = tk.Label(self.folder_control_frame, text='Horizontal angle N/A', font=('system', 8))
        self.status_horizontal.grid(row=2, column=0, sticky='W')
        
        self.status_vertical = tk.Label(self.folder_control_frame, text='Vertical angle N/A', font=('system', 8))
        self.status_vertical.grid(row=3, column=0, sticky='W')
        


        # Selecting the specimen 
        tk.Label(self.leftside_frame, text='Specimens').grid(row=3, column=0)
        self.specimen_box = Listbox(self.leftside_frame, ['(select directory)'], self.on_specimen_selection)
        self.specimen_box.grid(row=4, column=0, sticky='NS')

       
        # Selecting the recording
        tk.Label(self.leftside_frame, text='Image folders').grid(row=3, column=1)
        self.recording_box = Listbox(self.leftside_frame, [''], self.on_recording_selection)
        self.recording_box.grid(row=4, column=1, sticky='NS')


        # RIGHTSIDE frame        
        self.rightside_frame = tk.Frame(self)
        self.rightside_frame.grid(row=0, column=1, sticky='NS')
        
        
        canvas_constructor = lambda parent: CanvasPlotter(parent, visibility_button=False)
        tab_names = ['ROI', 'Displacement', 'XY']
        self.tabs = Tabs(self.rightside_frame, tab_names, [canvas_constructor for i in range(len(tab_names))])
        self.tabs.grid(row=0, column=1, sticky='NWES')

        # Make canvas plotter to stretch
        #self.tabs.grid_rowconfigure(1, weight=1)
        #self.tabs.grid_columnconfigure(0, weight=1)

        
        self.data_directory = None
        self.current_specimen = None
        self.selected_recording = None

        self.canvases = self.tabs.get_elements()
       

        self.default_button_bg = self.button_rois.cget('bg')

        self.plotter = RecordingPlotter()
                


    def set_data_directory(self):
        '''
        When the button set data diorectory is pressed.
        '''
        directory = filedialog.askdirectory(initialdir='/home/joni/smallbrains-nas1/array1')
        
        if directory:
            self.data_directory = directory
            self.core.set_data_directory(directory)
            
            specimens = self.core.list_specimens()

            self.specimen_box.set_selections(specimens)
    
    def copy_to_csv(self):
        
        os.makedirs(os.path.join(PROCESSING_TEMPDIR, self.current_specimen), exist_ok=True)
        
        with open(os.path.join(PROCESSING_TEMPDIR, self.current_specimen, self.selected_recording), 'w') as fp:
            
            for i_frame in range(len(self.plotter.magnitudes[0])):
                formatted = ','.join([str(self.plotter.magnitudes[i_repeat][i_frame]) for i_repeat in range(len(self.plotter.magnitudes)) ]) + '\n'
                fp.write(formatted)

    
    def copy_mean(self):
        '''
        
        '''
        formatted = '' 
        data = []

        self.root.clipboard_append(formatted)
        
        for pos_folder in self.analyser.list_imagefolders():
            all_movements = self.analyser.get_movements_from_folder(pos_folder)
            
            #movements = movements['left'] + movements['right']
            
            for eye, movements in all_movements.items():
                for repetition in range(len(movements)):
                    mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                    data.append(mag)
        
        for i_frame in range(len(data[0])):
            formatted += '\t'.join([str(data[i_repeat][i_frame]) for i_repeat in range(len(data)) ]) + '\n'
        self.root.clipboard_append(formatted)
        

    def copy_to_clipboard(self):
        self.root.clipboard_clear()
        
        formatted = ''
        

        #for points in self.plotter.magnitudes:
        #    formatted += '\t'.join([str(x) for x in points.tolist()]) + '\n'
        
        for i_frame in range(len(self.plotter.magnitudes[0])):
            formatted += '\t'.join([str(self.plotter.magnitudes[i_repeat][i_frame]) for i_repeat in range(len(self.plotter.magnitudes)) ]) + '\n'
        
        self.root.clipboard_append(formatted)
        
        self.copy_to_csv()

    def select_roi(self):
        '''
        Select/reselect a single ROI
        '''
        self.analyser.selectROIs(callback_on_exit=self.update_specimen,
                reselect_fns=[self.selected_recording], old_markings=True)


    def select_rois(self):
        '''
        When the analyse_button is pressed.
        '''

        # Ask confirmation if ROIs already selected
        if self.analyser.is_rois_selected():
            sure = messagebox.askokcancel('Reselect ROIs', 'Are you sure you want to reselect ROIs?')
            if not sure:
                return None

        self.analyser.selectROIs(callback_on_exit=self.update_specimen)


    def measure_movement(self):
        '''
        When the measure movement button is pressed.
        '''
        
        # Ask confirmation if ROIs already selected
        if self.analyser.is_measured():
            sure = messagebox.askokcancel('Remeasure movements', 'Are you sure you want to remeasure?')
            if not sure:
                return None
        
        MeasurementWindow(self.root, [self.analyser.measure_both_eyes], title='Measure movement')
    

    def antenna_level(self):
        '''
        Start antenna level search for the current specimen 
        '''
        
        #fullpath = os.path.join(self.data_directory, self.current_specimen)
        self.core.adm_subprocess('current', 'antenna_level', open_terminal=True)


    def update_specimen(self):
        self.on_specimen_selection(self.current_specimen)


    def on_specimen_selection(self, specimen):
        '''
        When a selection happens in the specimens listbox.
        '''
        self.current_specimen = specimen
        self.specimen_control_frame.config(text=specimen)

        self.analyser = self.core.get_manalyser(specimen)
        self.core.set_current_specimen(specimen)
        
        # Recordings box
        recordings = self.analyser.list_imagefolders()
        self.recording_box.enable()
        self.recording_box.set_selections(recordings)
        
        
        # Logick to set buttons inactive/active and their texts
        if self.analyser.is_rois_selected():
            

  
            self.button_rois.config(text='Reselect ROIs')
            self.button_rois.config(bg=self.default_button_bg)
            
            self.button_measure.config(state=tk.NORMAL)
            
            self.button_one_roi.config(state=tk.NORMAL)
           

            if self.analyser.is_measured():
                self.button_measure.config(text='Remeasure movement')
                self.button_measure.config(bg=self.default_button_bg)
            else:
                self.button_measure.config(text='Measure movement')
                self.button_measure.config(bg='green')
        else:
            self.recording_box.disable()

            self.button_rois.config(text='Select ROIs')
            self.button_rois.config(bg='green')

            self.button_measure.config(state=tk.DISABLED)
            self.button_measure.config(text='Measure movement')
            self.button_measure.config(bg=self.default_button_bg)

            self.button_one_roi.config(state=tk.DISABLED)

        if self.analyser.is_rois_selected():
            self.analyser.loadROIs()

        # Loading cached analyses and setting the recordings listbox

        if self.analyser.is_measured():
            self.analyser.load_analysed_movements()
            self.recording_box.enable()
            
        
        self.menu.update_states(self.analyser)

        self.plotter.set_analyser(self.analyser)
        
        N_rois = self.analyser.count_roi_selected_folders()
        N_image_folders = len(self.analyser.list_imagefolders())
        self.status_rois.config(text='ROIs selected {}/{}'.format(N_rois, N_image_folders))
        
        try:
            self.correction = self.analyser.get_antenna_level_correction()
        except:
            self.correction = False
        if self.correction:
            self.status_antenna_level.config(text='Zero corrected, {:.2f} degrees'.format(self.correction))
        else:
            self.status_antenna_level.config(text='Zero corrected FALSE')

        self.button_rois.config(state=tk.NORMAL)


    def on_recording_selection(self, selected_recording):
        '''
        When a selection happens in the recordings listbox.
        '''
        self.selected_recording = selected_recording

        
        angles = [list(angles_from_fn(selected_recording))]
        toDegrees(angles)
        horizontal, vertical = angles[0]
        self.status_horizontal.config(text='Horizontal angle {:.2f} degrees'.format(horizontal))
        self.status_vertical.config(text='Vertical angle {:.2f} degrees'.format(vertical))
        


        # Plotting related

        self.plotter.set_recording(selected_recording)


        i_plot = -1
        

        # Add imaging parameters
        i_plot += 1
        fig, ax = self.canvases[i_plot].get_figax()
        #ax.clear()
        self.plotter.ROI(ax)
       
        # Manipulating figure size? Put it to fill the window
        #fig.set_size_inches(1, 1, forward=True)
        #self.canvases[0].update_size()

        i_plot += 1
        fig, ax = self.canvases[i_plot].get_figax()
        ax.clear()
        self.plotter.magnitude(ax)
        
          
        i_plot += 1
        fig, ax = self.canvases[i_plot].get_figax()
        ax.clear()
        self.plotter.xy(ax) 

        
        for canvas in self.canvases:
            canvas.update()
 


class RecordingPlotter:

    def __init__(self):
        self.colorbar = None
        self.roi_rectangles = []


    def set_analyser(self, analyser):
        self.analyser = analyser


    def set_recording(self, selected_recording):
        self.selected_recording = selected_recording
        if self.analyser.is_measured():
            self.movement_data = self.analyser.get_movements_from_folder(selected_recording)
        else:
            self.movement_data = {}

    

    def magnitude(self, ax):

        self.magnitudes = []
        
        for eye, movements in self.movement_data.items():
            for repetition in range(len(movements)):
                mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                ax.plot(mag)
                
                self.magnitudes.append(mag)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Magnitude sqrt(x^2+y^2) (pixels)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

 
    def xy(self, ax):
        for eye, movements in self.movement_data.items():
            for repetition in range(len(movements)):
                
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
                self.colorbar.set_clim(0, N-1)
            
        ax.set_xlabel('Displacement in X (pixels)')
        ax.set_ylabel('Displacement in Y (pixels)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    def ROI(self, ax):
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
            old_roi.remove()
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

        text = '\n'.join(self.analyser.get_imaging_parameters(self.selected_recording))
        
        try:
            self.roi_text.set_text(text)
        except AttributeError:
            self.roi_text = self.roi_ax.text(0,1, text, ha='left', va='top', fontsize='small', 
                    transform=self.roi_ax.transAxes)

def main():
    
    if 'win' in sys.platform:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)

    root = tk.Tk()
    root.title('Pupil analysis - Tkinter GUI')
    ExamineView(root).grid()
    root.mainloop()


if __name__ == "__main__":
    main()

