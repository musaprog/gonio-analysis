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
import sys
import ctypes
import itertools

import numpy as np
import tkinter as tk
from tkinter import filedialog

from tk_steroids.routines import inspect_booleans
from tk_steroids.elements import (
        Listbox,
        Tabs,
        ButtonsFrame,
        ColorExplanation,
        TickboxFrame
        )
from tk_steroids.matplotlib import CanvasPlotter

from gonioanalysis import __version__
from gonioanalysis.directories import PROCESSING_TEMPDIR, GONIODIR
from gonioanalysis.rotary_encoders import to_degrees
from gonioanalysis.drosom.loading import angles_from_fn
from gonioanalysis.drosom.plotting.common import save_3d_animation
from gonioanalysis.drosom.plotting.basics import (
        plot_1d_magnitude, 
        plot_xy_trajectory,
        plot_3d_vectormap,
        )
from gonioanalysis.drosom.analyser_commands import ANALYSER_CMDS
from gonioanalysis.tkgui.core import Core
from gonioanalysis.tkgui.plotting import RecordingPlotter
from gonioanalysis.tkgui.widgets import RepetitionSelector

from gonioanalysis.tkgui.menu_commands import (
        ModifiedMenuMaker,
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
        
        #    Submenu: Add terminal commands
        self.terminal_commands = ModifiedMenuMaker(self.parent, self.core, 'Terminal interface commands')
        for name in ANALYSER_CMDS:
            setattr(self.terminal_commands, name, lambda name=name: self.core.adm_subprocess('current', "-A "+name) )
        self.terminal_commands._connect(self.specimen_commands.tkmenu)

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

        self.last_saveplotter_dir = GONIODIR

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
                ['Select ROIs', 'Measure movement'],
                [self.menu.specimen_commands.select_ROIs, self.menu.specimen_commands.measure_movement])
        self.buttons_frame_2.grid(row=1, column=0, sticky='NW', columnspan=2)
        self.button_rois, self.button_measure = self.buttons_frame_2.get_buttons()
        
        # Subframe for 2nd buttons frame
        #self.status_frame = tk.Frame(self.leftside_frame)
        #self.status_frame.grid(row=2)

        self.status_rois = tk.Label(self.specimen_control_frame, text='ROIs selected 0/0', font=('system', 8))
        self.status_rois.grid(row=2, column=0, sticky='W')
        
        self.status_antenna_level = tk.Label(self.specimen_control_frame, text='Zero correcter N/A', font=('system', 8))
        #self.status_antenna_level.grid(row=3, column=0, sticky='W')
        
        self.status_active_analysis = tk.Label(self.specimen_control_frame, text='Active analysis: default', font=('system', 8), justify=tk.LEFT)
        self.status_active_analysis.grid(row=4, column=0, sticky='W')
       
        self.tickbox_analyses = TickboxFrame(self.specimen_control_frame, [], ncols=4)
        self.tickbox_analyses.grid(row=5, column=0, sticky='W')


        # Image folder manipulations
        self.folder_control_frame = tk.LabelFrame(self.leftside_frame, text='Image folder')
        self.folder_control_frame.grid(row=2, column=0, sticky='NWES', columnspan=2)
       
        self.buttons_frame_3 = ButtonsFrame(self.folder_control_frame,
                ['Reselect ROI', 'Remeasure'],
                [self.menu.imagefolder_commands.select_ROIs, self.menu.imagefolder_commands.measure_movement])
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
        
        
        tab_kwargs = [{}, {}, {}, {'projection': '3d'}]
        tab_names = ['ROI', 'Displacement', 'XY', '3D']
        canvas_constructors = [lambda parent, kwargs=kwargs: CanvasPlotter(parent, visibility_button=False, **kwargs) for kwargs in tab_kwargs]
        self.tabs = Tabs(self.rightside_frame, tab_names, canvas_constructors,
                on_select_callback=self.update_plot)
        
        self.tabs.grid(row=0, column=0, sticky='NWES')


        # Make canvas plotter to stretch
        self.rightside_frame.grid_rowconfigure(0, weight=1)
        self.rightside_frame.grid_columnconfigure(0, weight=1)


        self.canvases = self.tabs.get_elements()
        
        # Controls for displacement plot (means etc)
        displacementplot_options, displacementplot_defaults = inspect_booleans(
                plot_1d_magnitude, exclude_keywords=['mean_imagefolders'])
        self.displacement_ticks = TickboxFrame(self.canvases[1], displacementplot_options,
                defaults=displacementplot_defaults, callback=lambda:self.update_plot(1))
        self.displacement_ticks.grid()

        xyplot_options, xyplot_defaults = inspect_booleans(
                plot_xy_trajectory)
        self.xy_ticks = TickboxFrame(self.canvases[2], xyplot_options,
                defaults=xyplot_defaults, callback=lambda:self.update_plot(2))
        self.xy_ticks.grid()



        # Controls for the vector plot
        # Controls for displacement plot (means etc)
        vectorplot_options, vectorplot_defaults = inspect_booleans(
                plot_3d_vectormap, exclude_keywords=[])
        self.vectorplot_ticks = TickboxFrame(self.canvases[3], vectorplot_options,
                defaults=vectorplot_defaults, callback=lambda:self.update_plot(3))
        self.vectorplot_ticks.grid()
        
        tk.Button(self.canvases[3], text='Save animation', command=self.save_3d_animation).grid()


        self.default_button_bg = self.button_rois.cget('bg')

        self.plotter = RecordingPlotter(self.core)
                
        # Add buttons for selecting single repeats from a recording
        self.repetition_selector = RepetitionSelector(self.rightside_frame, self.plotter, self.core,
                update_command=lambda: self.on_recording_selection('current'))
        self.repetition_selector.grid(row=1, column=0)
        

        tk.Button(self.repetition_selector, text='Copy data',
                command=self.copy_plotter_to_clipboard).grid(row=0, column=5)

        tk.Button(self.repetition_selector, text='Save view...',
                command=self.save_plotter_view).grid(row=0, column=6)




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


    def save_3d_animation(self):
        
        def callback():
            self.canvases[3].update()
        
        fig, ax = self.canvases[3].get_figax()
        save_3d_animation(self.core.analyser, ax=ax, interframe_callback=callback)


    def copy_to_csv(self, formatted): 
        with open(os.path.join(GONIODIR, 'clipboard.csv'), 'w') as fp:
            fp.write(formatted.replace('\t', ','))


    def specimen_traces_to_clipboard(self, mean=False):
        '''
        If mean==True, copy only the average trace.
        Otherwise, copy all the traces of the fly.
        '''

        formatted = '' 
        
        # Always first clear clipboard; If something goes wrong, the user
        # doesn't want to keep pasting old data thinking it's new.
        self.root.clipboard_clear()

        if self.core.selected_recording is None:
            return None

        data = []

        self.root.clipboard_append(formatted)
        
        for pos_folder in self.core.analyser.list_imagefolders():
            all_movements = self.core.analyser.get_movements_from_folder(pos_folder)
            
            for eye, movements in all_movements.items():
                for repetition in range(len(movements)):
                    mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                    data.append(mag)

        if mean:
            data = [np.mean(data, axis=0)]
        
        for i_frame in range(len(data[0])):
            formatted += '\t'.join([str(data[i_repeat][i_frame]) for i_repeat in range(len(data)) ]) + '\n'
        
        self.root.clipboard_append(formatted)
        self.copy_to_csv(formatted)       



    def copy_plotter_to_clipboard(self, force_i_tab=None):
        '''
        Copies data currently visible on theopen plotter tab to the clipboard.
        
        force_i_tab         Copy from the specified tab index, instead of
                            the currently opened tab
        '''
        formatted = ''
        
        # Always first clear clipboard; If something goes wrong, the user
        # doesn't want to keep pasting old data thinking it's new.
        self.root.clipboard_clear()
        
        if self.core.selected_recording is None:
            return None

        if force_i_tab is not None:
            i_tab = int(force_i_tab)
        else:
            i_tab = self.tabs.i_current
        
        # Make sure we have the correct data in the plot by reissuing
        # the plotting command
        self.update_plot(i_tab)
        
        # Select data based on where we want to copy
        if i_tab == 0:
            data = self.plotter.image
        elif i_tab == 1:
            data = self.plotter.magnitudes
        elif i_tab == 2:
            data = self.plotter.xys
            data = list(itertools.chain(*data))
        elif i_tab == 3:
            raise NotImplementedError('Cannot yet cliboard vectormap data')

        # Format the data for tkinter clipboard copy
        for i_frame in range(len(data[0])):
            formatted += '\t'.join([str(data[i_repeat][i_frame]) for i_repeat in range(len(data)) ]) + '\n'
        
        self.root.clipboard_append(formatted)
        self.copy_to_csv(formatted)


    
    def save_plotter_view(self):
        '''
        Launches a save dialog for the current plotter view.
        '''
        fig, ax = self.canvases[self.tabs.i_current].get_figax()
        
        dformats = fig.canvas.get_supported_filetypes()
        formats = [(value, '*.'+key) for key, value in sorted(dformats.items())]
        
        # Make png first
        if 'png' in dformats.keys():
            i = formats.index((dformats['png'], '*.png'))
            formats.insert(0, formats.pop(i))

        fn = filedialog.asksaveasfilename(title='Save current view',
                initialdir=self.last_saveplotter_dir,
                filetypes=formats)

        if fn:
            self.last_saveplotter_dir = os.path.dirname(fn)

            fig.savefig(fn, dpi=1200)
            
        


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

        self.status_active_analysis.config(text='Active analysis: {}'.format(self.core.analyser.active_analysis))

        
        # FIXME Instead of destroyign tickbox, make changes to tk_steroids
        # so that the selections can be reset
        self.tickbox_analyses.grid_forget()
        self.tickbox_analyses.destroy()
        self.tickbox_analyses = TickboxFrame(self.specimen_control_frame, self.core.analyser.list_analyses(),
                defaults=[self.core.analyser.active_analysis == an for an in self.core.analyser.list_analyses()],
                ncols=4, callback=lambda: self.update_plot(None))
        self.tickbox_analyses.grid(row=5, column=0, sticky='W')

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
        

        # Plotting only the view we have currently open
        self.update_plot(self.tabs.i_current)
        


    def update_plot(self, i_plot):
        '''
        i_plot : int or None
            Index of the plot (from 0 to N-1 tabs) or None just to update
        '''
        if self.core.selected_recording is None:
            return None
        
        if i_plot is None:
            i_plot = self.tabs.i_current

        fig, ax = self.canvases[i_plot].get_figax()

        if i_plot == 0:
            self.plotter.ROI(ax)
        else:
            
            ax.clear()

            remember_analysis = self.core.analyser.active_analysis

            for analysis in [name for name, state in self.tickbox_analyses.states.items() if state == True]:
                
                self.core.analyser.active_analysis = analysis

                if i_plot == 1:
                    self.plotter.magnitude(ax, **self.displacement_ticks.states)
                elif i_plot == 2:  
                    self.plotter.xy(ax, **self.xy_ticks.states) 
                elif i_plot == 3:
                    self.plotter.vectormap(ax, **self.vectorplot_ticks.states)
            
            self.core.active_analysis = remember_analysis


        self.canvases[i_plot].update()
        
        self.repetition_selector.update_text()


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
    root.title('Gonio analysis - Tkinter GUI - {}'.format(__version__))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.minsize(800,600)
    ExamineView(root).grid(sticky='NSWE')
    root.mainloop()


if __name__ == "__main__":
    main()

