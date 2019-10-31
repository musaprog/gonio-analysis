

import tkinter as tk


from tk_steroids.elements import Listbox
from tk_steroids.matplotlib import CanvasPlotter

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
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        tk.Button(self, text='Set data directory...', command=self.set_data_directory).grid(row=0, column=0)
        
        self.specimen_box = Listbox(self, ['fly1', 'fly2'], print)
        self.specimen_box.grid(row=1, column=0, sticky='NS')
        
        self.recording_box = Listbox(self, ['rot1', 'rot2'], print)
        self.recording_box.grid(row=1, column=1, sticky='NS')
        
        CanvasPlotter(self).grid(row=0, column=2, rowspan=3)


    def set_data_directory(self):
        
        self.specimen_box.set_selections(['testA'])
    
    def on_specimen_selection(specimen):
        pass

def main():
    root = tk.Tk()
    ExamineView(root).grid()
    root.mainloop()


if __name__ == "__main__":
    main()

