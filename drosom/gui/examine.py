
import os

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog


from tk_steroids.elements import Listbox, Tabs, ButtonsFrame
from tk_steroids.matplotlib import CanvasPlotter

from pupil.drosom.analysing import MAnalyser
from pupil.drosom.gui.run_measurement import MeasurementWindow


class ExamineMenubar(tk.Frame):
    '''
    Menubar class for the examine GUI.
    '''

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Set data directory', command=self.batch_ROIs)
        file_menu.add_command(label='Exit', command=self.on_exit)
        self.menubar.add_cascade(label='File', menu=file_menu)
        
        # Batch run
        batch_menu = tk.Menu(self.menubar, tearoff=0)
        batch_menu.add_command(label='Select ROIs', command=self.batch_ROIs)
        batch_menu.add_command(label='Measure movements', command=self.batch_measurement)
        self.menubar.add_cascade(label='Batch', menu=batch_menu)
        
        
        # Data plotting
        plot_menu = tk.Menu(self.menubar, tearoff=0)
        plot_menu.add_command(label='Vectormap', command=self.batch_ROIs)
        plot_menu.add_command(label='Averaged vectormap...', command=self.batch_measurement)
        self.menubar.add_cascade(label='Plot', menu=plot_menu)
        

        self.winfo_toplevel().config(menu=self.menubar)
    
    def batch_ROIs(self):
        pass

    def batch_measurement(self):
        pass

    
    def on_exit(self):
        self.winfo_toplevel().destroy()
    
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
        
        #tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        # Uncomment to have the menubar
        #self.menu = ExamineMenubar(self)

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
        self.specimen_box = Listbox(self.leftside_frame, ['fly1', 'fly2'], self.on_specimen_selection)
        self.specimen_box.grid(row=2, column=0, sticky='NS')

       
        # Selecting the recording
        self.recording_box = Listbox(self.leftside_frame, ['rot1', 'rot2'], self.on_recording_selection)
        self.recording_box.grid(row=2, column=1, sticky='NS')


        # RIGHTSIDE frame        
        self.rightside_frame = tk.Frame(self)
        self.rightside_frame.grid(row=0, column=1, sticky='NS')
        
        
        canvas_constructor = lambda parent: CanvasPlotter(parent, visibility_button=False)
        self.tabs = Tabs(self.rightside_frame, ['XY', 'Magnitude'], [canvas_constructor, canvas_constructor])
        self.tabs.grid(row=0, column=1)

        self.canvases = self.tabs.get_elements()
       

        self.default_button_bg = self.button_rois.cget('bg')


    def set_data_directory(self):
        '''
        When the button set data diorectory is pressed.
        '''
        print(dir(filedialog.askdirectory))
        directory = filedialog.askdirectory(initialdir='/home/joni/smallbrains-nas1/array1')
        
        if directory:
            self.data_directory = directory
            
            specimens = [fn for fn in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, fn))]

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
        self.analyser = MAnalyser(self.data_directory, specimen)

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
   

    def on_recording_selection(self, selected_recording):
        '''
        When a selection happens in the recordings listbox.
        '''
        movement_data = self.analyser.get_movements_from_folder(selected_recording)
        
        print(movement_data)
        
        
        fig, ax = self.canvases[0].get_figax()
        ax.clear()
        
        for eye, movements in movement_data.items():
            for repetition in range(len(movements)):
                x = movements[repetition]['x']
                y = movements[repetition]['y']
                ax.plot(x, y)

        
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
    ExamineView(root).grid()
    root.mainloop()


if __name__ == "__main__":
    main()

