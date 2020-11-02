'''

TODO

Features
+ video of the moving ROIs
+ no need for manual add to PythonPath on Windows
- window to strecth to full screen
- multiselect ROIs (if many separate recordings at the same site)

Polishing
- specimens control title, retain name specimen
- image folders control title as specimens
- after movement measure, update MAnalyser
- displacement plot Y axis label
- highlight specimens/image_folders:
    red: no rois / movements
    yellow: rois but no movements
- x-axis from frames to time?


'''
import os
import multiprocessing
import subprocess
import sys
import ctypes
import json

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

# PLotting
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.widgets
import tifffile

from tk_steroids.elements import Listbox, Tabs, ButtonsFrame, TickSelect, ColorExplanation
from tk_steroids.matplotlib import CanvasPlotter
from tk_steroids.menumaker import MenuMaker

from pupilanalysis.directories import PROCESSING_TEMPDIR, PROCESSING_TEMPDIR_BIGFILES
from pupilanalysis.antenna_level import AntennaLevelFinder
from pupilanalysis.rotary_encoders import to_degrees
from pupilanalysis.drosom.loading import angles_from_fn
from pupilanalysis.drosom.analysing import MAnalyser
from pupilanalysis.drosom.plotting import MPlotter
from pupilanalysis.drosom.gui.core import Core
from pupilanalysis.drosom.gui.plotting import RecordingPlotter
from pupilanalysis.drosom.gui.zero_correct import ZeroCorrect
from pupilanalysis.drosom.gui.repetition_selection import RepetitionSelector
from pupilanalysis.drosom.kinematics import mean_max_response
from pupilanalysis.drosom import linked_data

from pupilanalysis.drosom.gui.menu_commands import (
        FileCommands,
        ImageFolderCommands,
        SpecimenCommands,
        ManySpecimenCommands,
        OtherCommands
        )



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

        # File command and menu
        self.file_commands = FileCommands(self.parent, self.core, 'File')
        self.file_commands._connect(self.menubar, tearoff=0)
        
        self.menubar.add_command(label="|")

        # Imagefolder command and menu
        self.imagefolder_commands = ImageFolderCommands(self.parent, self.core, 'Image folder')
        self.imagefolder_commands._connect(self.menubar, tearoff=0)

        # Specimen commands and menu
        self.specimen_commands = SpecimenCommands(self.parent, self.core, 'Specimen')
        self.specimen_commands._connect(self.menubar, tearoff=0)

        # Many specimen commands and menu
        self.many_specimen_commands = ManySpecimenCommands(self.parent, self.core, 'Many specimens')
        self.many_specimen_commands._connect(self.menubar, tearoff=0)
       
        self.menubar.add_command(label="|")
        
        # Other commands and menu
        self.other_commands = OtherCommands(self.parent, self.core, 'Other')
        self.other_commands._connect(self.menubar, tearoff=0)
        

        self.winfo_toplevel().config(menu=self.menubar)
   
       


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
        self.core.update_gui = self.update_specimen
        
        self.root = self.winfo_toplevel()
        
        # Make canvas plotter to stretch
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(0, minsize=400)


        #tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        # Uncomment to have the menubar
        self.menu = ExamineMenubar(self)

        # LEFTSIDE frame
        self.leftside_frame = tk.Frame(self)
        self.leftside_frame.grid(row=0, column=0, sticky='NSWE') 
        self.leftside_frame.grid_rowconfigure(4, weight=1) 
        self.leftside_frame.grid_columnconfigure(0, weight=1)
        self.leftside_frame.grid_columnconfigure(1, weight=1)



        # The 1st buttons frame, selecting root data directory
        self.buttons_frame_1 = ButtonsFrame(self.leftside_frame, ['Set data directory'],
                [self.menu.file_commands.set_data_directory, self.menu.file_commands.set_data_directory])
        self.buttons_frame_1.grid(row=0, column=0, sticky='NW', columnspan=2)

 
        self.specimen_control_frame = tk.LabelFrame(self.leftside_frame, text='Specimen')
        self.specimen_control_frame.grid(row=1, column=0, sticky='NWES', columnspan=2)

       
        # The 2nd buttons frame, ROIs and movements
        self.buttons_frame_2 = ButtonsFrame(self.specimen_control_frame,
                ['Select ROIs', 'Measure movement', 'Zero correct', 'Copy to clipboard'],
                [self.menu.specimen_commands.select_ROIs, self.menu.specimen_commands.measure_movement, self.menu.specimen_commands.antenna_level, self.specimen_traces_to_clipboard])
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
                ['Reselect ROI', 'Remeasure', 'Copy to clipboard'],
                [self.menu.imagefolder_commands.select_ROIs, self.menu.imagefolder_commands.measure_movement, self.copy_to_clipboard])
        self.buttons_frame_3.grid(row=1, column=0, sticky='NW', columnspan=2)
        self.button_one_roi = self.buttons_frame_2.get_buttons()[0]
        

        self.status_horizontal = tk.Label(self.folder_control_frame, text='Horizontal angle N/A', font=('system', 8))
        self.status_horizontal.grid(row=2, column=0, sticky='W')
        
        self.status_vertical = tk.Label(self.folder_control_frame, text='Vertical angle N/A', font=('system', 8))
        self.status_vertical.grid(row=3, column=0, sticky='W')
        


        # Selecting the specimen 
        tk.Label(self.leftside_frame, text='Specimens').grid(row=3, column=0)
        self.specimen_box = Listbox(self.leftside_frame, ['(select directory)'], self.on_specimen_selection)
        self.specimen_box.grid(row=4, column=0, sticky='NSEW')

       
        # Selecting the recording
        tk.Label(self.leftside_frame, text='Image folders').grid(row=3, column=1)
        self.recording_box = Listbox(self.leftside_frame, [''], self.on_recording_selection)
        self.recording_box.grid(row=4, column=1, sticky='NSEW')

        
        # Add color explanation frame in the bottom
        ColorExplanation(self.leftside_frame, ['white', 'green', 'yellow'],
                ['Movements measured', 'ROIs selected', 'No ROIs']).grid(row=5, column=0, sticky='NW')

        # RIGHTSIDE frame        
        self.rightside_frame = tk.Frame(self)
        self.rightside_frame.grid(row=0, column=1, sticky='NWES')
        
        
        canvas_constructor = lambda parent: CanvasPlotter(parent, visibility_button=False)
        tab_names = ['ROI', 'Displacement', 'XY']
        self.tabs = Tabs(self.rightside_frame, tab_names, [canvas_constructor for i in range(len(tab_names))])
        self.tabs.grid(row=0, column=0, sticky='NWES')

        # Make canvas plotter to stretch
        self.rightside_frame.grid_rowconfigure(0, weight=1)
        self.rightside_frame.grid_columnconfigure(0, weight=1)


        self.canvases = self.tabs.get_elements()
       

        self.default_button_bg = self.button_rois.cget('bg')

        self.plotter = RecordingPlotter()
                
        # Add buttons for selecting single repeats from a recording
        RepetitionSelector(self.rightside_frame, self.plotter, update_command=lambda: self.on_recording_selection('current')).grid(row=1, column=0)




    def _color_specimens(self, specimens):
        '''
        See _color_recording for reference.
        '''
        colors = []
        for specimen in specimens:
            analyser = self.core.get_manalyser(specimen, no_data_load=True)
            color = 'yellow'
            if analyser.are_rois_selected():
                color = 'green'
                if analyser.is_measured():
                    color = 'white'
            
            colors.append(color)
        return colors




    def copy_to_csv(self):
        
        directory = os.path.join(PROCESSING_TEMPDIR, 'clipboard', self.core.current_specimen)
        os.makedirs(directory, exist_ok=True)
        
        with open(os.path.join(directory, self.core.selected_recording+'.csv'), 'w') as fp:
            
            for i_frame in range(len(self.plotter.magnitudes[0])):
                formatted = ','.join([str(self.plotter.magnitudes[i_repeat][i_frame]) for i_repeat in range(len(self.plotter.magnitudes)) ]) + '\n'
                fp.write(formatted)

    



    def specimen_traces_to_clipboard(self, mean=False):
        '''
        If mean==True, copy only the average trace.
        Otherwise, copy all the traces of the fly.
        '''
        formatted = '' 
        data = []

        self.root.clipboard_clear()
        
        self.root.clipboard_append(formatted)
        
        for pos_folder in self.core.analyser.list_imagefolders():
            all_movements = self.core.analyser.get_movements_from_folder(pos_folder)
            
            #movements = movements['left'] + movements['right']
            
            for eye, movements in all_movements.items():
                for repetition in range(len(movements)):
                    mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                    data.append(mag)

        if mean:
            data = [np.mean(data, axis=0)]
        
        for i_frame in range(len(data[0])):
            formatted += '\t'.join([str(data[i_repeat][i_frame]) for i_repeat in range(len(data)) ]) + '\n'
        self.root.clipboard_append(formatted)
       


    def copy_data_to_clipboard(self, data):
        self.root.clipboard_clear()

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
       
        

    def _color_recordings(self, recordings):
        '''
        Returns a list of colours, each corresponding to a recording
        in recordings, based on wheter the ROIs have been selected or
        movements measured for the recording.

        yellow      No ROIs, no movements
        green       ROIs, no movements
        white       ROIs and movements
        '''
        colors = []
        for recording in recordings:
            color = 'yellow'
            if self.core.analyser.folder_has_rois(recording):
                color = 'green'
                if self.core.analyser.folder_has_movements(recording):
                    color = 'white'
            
            colors.append(color)
        return colors


    def on_specimen_selection(self, specimen):
        '''
        When a selection happens in the specimens listbox.
        '''
        self.specimen_control_frame.config(text=specimen)

        self.core.set_current_specimen(specimen)
        

        # Recordings box
        recordings = self.core.analyser.list_imagefolders()
        self.recording_box.enable()
        self.recording_box.set_selections(recordings, colors=self._color_recordings(recordings))
         
        
        # Logick to set buttons inactive/active and their texts
        if self.core.analyser.are_rois_selected(): 
  
            self.button_rois.config(text='Reselect ROIs')
            self.button_rois.config(bg=self.default_button_bg)
            
            self.button_measure.config(state=tk.NORMAL)
            
            self.button_one_roi.config(state=tk.NORMAL)
            
            # Enable image_folder buttons
            for button in self.buttons_frame_3.get_buttons():
                button.config(state=tk.NORMAL)

            if self.core.analyser.is_measured():
                self.button_measure.config(text='Remeasure movement')
                self.button_measure.config(bg=self.default_button_bg)
            else:
                self.button_measure.config(text='Measure movement')
                self.button_measure.config(bg='green')
        else:
            #self.recording_box.disable()

            self.button_rois.config(text='Select ROIs')
            self.button_rois.config(bg='yellow')

            self.button_measure.config(state=tk.DISABLED)
            self.button_measure.config(text='Measure movement')
            self.button_measure.config(bg=self.default_button_bg)

            self.button_one_roi.config(state=tk.DISABLED)
            
            # Disable image_folder buttons
            for button in self.buttons_frame_3.get_buttons():
                button.config(state=tk.DISABLED)



        if self.core.analyser.are_rois_selected():
            self.core.analyser.load_ROIs()

        # Loading cached analyses and setting the recordings listbox

        if self.core.analyser.is_measured():
            self.core.analyser.load_analysed_movements()
            #self.recording_box.enable()
            
        
        self.plotter.set_analyser(self.core.analyser)
        
        N_rois = self.core.analyser.count_roi_selected_folders()
        N_image_folders = len(self.core.analyser.list_imagefolders())
        self.status_rois.config(text='ROIs selected {}/{}'.format(N_rois, N_image_folders))
        
        try:
            self.correction = self.core.analyser.get_antenna_level_correction()
        except:
            self.correction = False
        if self.correction is not False:
            self.status_antenna_level.config(text='Zero corrected, {:.2f} degrees'.format(self.correction))
        else:
            self.status_antenna_level.config(text='Zero corrected FALSE')

        self.button_rois.config(state=tk.NORMAL)


    def on_recording_selection(self, selected_recording):
        '''
        When a selection happens in the recordings listbox.
        
        selected_recording      Name of the recording. If 'current', keeps the current
        '''
        if selected_recording == 'current':
            selected_recording = self.core.selected_recording
        else:
            self.core.set_selected_recording(selected_recording)

        
        print(self.core.analyser.get_recording_time(selected_recording))
        
        angles = [list(angles_from_fn(selected_recording))]
        to_degrees(angles)
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
        #fig.set_size_inches(10, 10, forward=True)
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


    def update_specimen(self, changed_specimens=False):
        '''
        Updates GUI colors, button states etc. to right values.
        
        Call this if there has been changes to specimens/image_folders by an
        external process or similar.
        '''        
        if changed_specimens:
            specimens = self.core.list_specimens()
            self.specimen_box.set_selections(specimens, self._color_specimens(specimens))

        if self.core.current_specimen is not None:
            self.on_specimen_selection(self.core.current_specimen)





def main():
    
    if 'win' in sys.platform:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)

    root = tk.Tk()
    root.title('Pupil analysis - Tkinter GUI')
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.minsize(800,600)
    ExamineView(root).grid(sticky='NSWE')
    root.mainloop()


if __name__ == "__main__":
    main()

