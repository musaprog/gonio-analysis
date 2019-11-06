
import os
import multiprocessing
import subprocess
import sys


import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

# PLotting
import matplotlib.pyplot as plt
import matplotlib.cm

from tk_steroids.elements import Listbox, Tabs, ButtonsFrame, TickSelect
from tk_steroids.matplotlib import CanvasPlotter

from pupil.drosom.analysing import MAnalyser
from pupil.drosom.plotting import MPlotter
from pupil.drosom.gui.run_measurement import MeasurementWindow
from pupil.drosom.gui.core import Core



class ExamineMenubar(tk.Frame):
    '''
    Menubar class for the examine GUI.
    '''

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.core = parent.core
        self.menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Set data directory', command=self.parent.set_data_directory)
        file_menu.add_command(label='Exit', command=self.on_exit)
        self.menubar.add_cascade(label='File', menu=file_menu)

        # Batch run
        batch_menu = tk.Menu(self.menubar, tearoff=0)
        batch_menu.add_command(label='Select ALL ROIs', command=self.batch_ROIs)
        batch_menu.add_command(label='Measure ALL movements', command=self.batch_measurement)
        self.menubar.add_cascade(label='Batch', menu=batch_menu)
        
        
        # Data plotting
        plot_menu = tk.Menu(self.menubar, tearoff=0)
        plot_menu.add_command(label='Vectormap', command=lambda: self.core.adm_subprocess('current', 'vectormap'))
        plot_menu.add_command(label='Vectormap - rotating video', command=lambda: self.core.adm_subprocess('current', 'vectormap animation')) 
        plot_menu.add_command(label='Averaged vectormap...', command=self.averaged_vectormap)
        self.menubar.add_cascade(label='Plot', menu=plot_menu)
        self.plot_menu = plot_menu        

        self.winfo_toplevel().config(menu=self.menubar)


    def batch_ROIs(self):
        pass


    def batch_measurement(self):
        pass


    def averaged_vectormap(self):
        top = tk.Toplevel()
        top.title('Select specimens')
        
        specimens = self.core.list_specimens() 
        selector = TickSelect(top, specimens, lambda specimens: self.core.adm_subprocess(specimens, 'averaged'))
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
        
        tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        # Uncomment to have the menubar
        self.menu = ExamineMenubar(self)

        # LEFTSIDE frame
        self.leftside_frame = tk.Frame(self)
        self.leftside_frame.grid(row=0, column=0, sticky='NS') 
        self.leftside_frame.grid_rowconfigure(2, weight=1) 

        # The 1st buttons frame, selecting root data directory
        self.buttons_frame_1 = ButtonsFrame(self.leftside_frame, ['Set data directory'],
                [self.set_data_directory, self.set_data_directory])
        self.buttons_frame_1.grid(row=0, column=0, sticky='NW', columnspan=2)
        
        # The 2nd buttons frame, ROIs and movements
        self.buttons_frame_2 = ButtonsFrame(self.leftside_frame,
                ['Select ROIs', 'Measure movement'],
                [self.select_rois, self.measure_movement])
        self.buttons_frame_2.grid(row=1, column=0, sticky='NW', columnspan=2)
        self.button_rois, self.button_measure = self.buttons_frame_2.get_buttons()


        # Selecting the specimen
        self.specimen_box = Listbox(self.leftside_frame, ['(select directory)'], self.on_specimen_selection)
        self.specimen_box.grid(row=2, column=0, sticky='NS')

       
        # Selecting the recording
        self.recording_box = Listbox(self.leftside_frame, [''], self.on_recording_selection)
        self.recording_box.grid(row=2, column=1, sticky='NS')


        # RIGHTSIDE frame        
        self.rightside_frame = tk.Frame(self)
        self.rightside_frame.grid(row=0, column=1, sticky='NS')
        
        
        canvas_constructor = lambda parent: CanvasPlotter(parent, visibility_button=False)
        tab_names = ['XY', 'Magnitude', 'ROI']
        self.tabs = Tabs(self.rightside_frame, tab_names, [canvas_constructor for i in range(len(tab_names))])
        self.tabs.grid(row=0, column=1)

        self.canvases = self.tabs.get_elements()
       

        self.default_button_bg = self.button_rois.cget('bg')

        
        self.colorbar = None


    def set_data_directory(self):
        '''
        When the button set data diorectory is pressed.
        '''
        print(dir(filedialog.askdirectory))
        directory = filedialog.askdirectory(initialdir='/home/joni/smallbrains-nas1/array1')
        
        if directory:
            self.core.set_data_directory(directory)
            
            specimens = self.core.list_specimens()

            self.specimen_box.set_selections(specimens)


    def select_rois(self):
        '''
        When the analyse_button is pressed.
        '''

        # Ask confirmation if ROIs already selected
        if self.analyser.is_rois_selected():
            sure = messagebox.askokcancel('Reselect ROIs', 'Are you sure you want to reselect ROIs?')
            if not sure:
                return None

        self.analyser.selectROIs()


    def measure_movement(self):
        '''
        When the measure movement button is pressed.
        '''
        
        # Ask confirmation if ROIs already selected
        if self.analyser.is_measured():
            sure = messagebox.askokcancel('Reselect ROIs', 'Are you sure you want to reselect ROIs?')
            if not sure:
                return None
        
        MeasurementWindow(self.analyser)


    def on_specimen_selection(self, specimen):
        '''
        When a selection happens in the specimens listbox.
        '''
        self.analyser = self.core.get_manalyser(specimen)
        self.core.set_current_specimen(specimen)

        # Logick to set buttons inactive/active and their texts
        if self.analyser.is_rois_selected():
            
            self.button_rois.config(text='Reselect ROIs')
            self.button_rois.config(bg=self.default_button_bg)
            self.button_measure.config(state=tk.NORMAL)
            

            if self.analyser.is_measured():
                self.button_measure.config(text='Remeasure movement')
                self.button_measure.config(bg=self.default_button_bg)
            else:
                self.button_measure.config(text='Measure movement')
                self.button_measure.config(bg='green')
        else:
            self.button_rois.config(text='Select ROIs')
            self.button_rois.config(bg='green')

            self.button_measure.config(state=tk.DISABLED)
            self.button_measure.config(text='Measure movement')
            self.button_measure.config(bg=self.default_button_bg)

        # Loading cached analyses and setting the recordings listbox
        if self.analyser.is_measured():
            self.analyser.load_analysed_movements()
            recordings = self.analyser.list_imagefolders()
        else:
            recordings = ['not analysed']
        self.recording_box.set_selections(recordings)
        
        self.menu.update_states(self.analyser)

    def on_recording_selection(self, selected_recording):
        '''
        When a selection happens in the recordings listbox.
        '''
        movement_data = self.analyser.get_movements_from_folder(selected_recording)
        
        
        fig, ax = self.canvases[0].get_figax()
        ax.clear()
        
        # Plot XY
        for eye, movements in movement_data.items():
            for repetition in range(len(movements)):
                
                x = movements[repetition]['x']
                y = movements[repetition]['y']
                 
                N = len(movements[repetition]['x'])
                
                cmap = matplotlib.cm.get_cmap('inferno', N)
               
                for i_point in range(1, N):
                    ax.plot([x[i_point-1], x[i_point]], [y[i_point-1], y[i_point]], color=cmap((i_point-1)/(N-1)))
                
        
        # Colormap
        
        if not self.colorbar: 
            time = [i for i in range(N)]
            sm = matplotlib.cm.ScalarMappable(cmap=cmap)
            sm.set_array(time)

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



        fig, ax = self.canvases[1].get_figax()
        ax.clear()
        
        for eye, movements in movement_data.items():
            for repetition in range(len(movements)):
                mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                ax.plot(mag)
        



        for canvas in self.canvases:
            canvas.update()



def main():
    root = tk.Tk()
    root.title('Pupil analysis - Tkinter GUI')
    ExamineView(root).grid()
    root.mainloop()


if __name__ == "__main__":
    main()

