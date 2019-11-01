
import os

import numpy as np
import tkinter as tk
import tkinter.filedialog as filedialog

from tk_steroids.elements import Listbox, Tabs
from tk_steroids.matplotlib import CanvasPlotter

from pupil.drosom.analysing import MAnalyser

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
        
        tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        self.specimen_box = Listbox(self, ['fly1', 'fly2'], self.on_specimen_selection)
        self.specimen_box.grid(row=1, column=0, sticky='NS')
        
        self.recording_box = Listbox(self, ['rot1', 'rot2'], self.on_recording_selection)
        self.recording_box.grid(row=1, column=1, sticky='NS')
       
        
        canvas_constructor = lambda parent: CanvasPlotter(parent, visibility_button=False)
        self.tabs = Tabs(self, ['XY', 'Magnitude'], [canvas_constructor, canvas_constructor])
        self.tabs.grid(row=0, column=2, rowspan=3)

        self.canvases = self.tabs.get_elements()
        
        #self.canvas = CanvasPlotter(self)
        #self.canvas.grid(row=0, column=2, rowspan=3)

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
        

    def on_specimen_selection(self, specimen):
        
        self.analyser = MAnalyser(self.data_directory, specimen)

        if self.analyser.is_analysed():
            
            self.analyser.load_analysed_movements()
            
            recordings = self.analyser.list_imagefolders()
        else:
            recordings = ['not analysed']
        
        self.recording_box.set_selections(recordings)
   

    def on_recording_selection(self, selected_recording):

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

