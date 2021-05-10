'''
Gonio-analysis tkinter GUI widgets.
'''
import os
import numpy as np
import threading

import tkinter as tk
import tifffile

from gonioanalysis.antenna_level import (
        load_drosom,
        save_antenna_level_correction,
        load_reference_fly
        )
from tk_steroids.matplotlib import CanvasPlotter


class WaitingFrame(tk.Frame):

    def __init__(self, tk_master, text):
        tk.Frame.__init__(self, tk_master)
        tk.Label(self, text=text).grid()


class WaitingWindow():
    '''
    Spans a new tkinter root window so use this only
    if there's no tkinter root beforehad.
    '''
    def __init__(self, title, text):
        
        self.root = tk.Tk()
        self.root.title(title)
        WaitingFrame(self.root, text).grid()
        self.root.update()

    def close(self):
        self.root.destroy()



class ZeroCorrect(tk.Frame):
    '''
    Creates a frame where one can perform zero correction (atenna level search)
    for a specimen using reference specimen (alr, antenna level reference)
    
    This is (maybe better) alternative to the command line tool, antenna_level.py,
    that uses binary search tactics to match.
    '''

    def __init__(self, tk_parent, specimen_path, alr_data_path, callback=None):
        tk.Frame.__init__(self, tk_parent)
        self.parent = tk_parent
        self.callback = callback

        self.specimen_name = os.path.basename(specimen_path)

        # Load data
        self.specimen_pitches, self.specimen_images = load_drosom(specimen_path)
        #self.reference_pitches, self.reference_images = {fn: pitch for pitch, fn in loadReferenceFly(alr_data_path).items()}
        
        self.reference_pitches, self.reference_images = [[],[]]
        for pitch, fn in sorted(load_reference_fly(alr_data_path).items(), key=lambda x: float(x[0])):
            self.reference_pitches.append(pitch)
            self.reference_images.append(fn)

        # Set plotters
        self.specimen_plotter = CanvasPlotter(self, text=specimen_path)
        self.specimen_plotter.grid(row=1, column=0, sticky='NSWE')

        self.reference_plotter = CanvasPlotter(self, text='Reference fly')
        self.reference_plotter.grid(row=1, column=1, sticky='NSEW')
      
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        

        # Help text
        tk.Label(self, text='Rotate the reference until it matches the specimen and press Next image.\nAlternatively, set manual correction.').grid(row=0, column=0, columnspan=2)

        # Set buttons
        buttons_frame = tk.LabelFrame(self, text='Rotate reference')
        buttons_frame.grid(row=2, column=0, columnspan=2)
        steps = [-20, -5, -3, -1, 1, 3, 5, 20]
        for i_column, step in enumerate(steps):
            button = tk.Button(buttons_frame, text=str(step), command=lambda step=step: self.rotate_reference(step))
            button.grid(row=1, column=i_column)
        
        self.set_button = tk.Button(self, text='Next image', command=self.set_image)
        self.set_button.grid(row=3, column=0, columnspan=2)
        
        self.set_manual_button = tk.Button(self, text='Set manual correction...', command=self.set_manual)
        self.set_manual_button.grid(row=3, column=1, sticky='E')
        

        # Loop variables
        self.i_specimen = 0
        self.i_reference = 0
    
        # Offset between each specimen-reference image is saved here.
        self.offsets = []
        
        self.update_plots() 


    def rotate_reference(self, steps):
        '''
        When user clicks to rotate the reference fly.
        '''
        self.i_reference += steps
        
        if self.i_reference >= len(self.reference_pitches):
            self.i_reference = len(self.reference_pitches) - 1
        elif self.i_reference < 0:
            self.i_reference = 0
        
        self.update_plots()
        

    def set_image(self):
        '''
        When user sets the current reference rotation as the best match
        '''
        offset = float(self.specimen_pitches[self.i_specimen]) - float(self.reference_pitches[self.i_reference])
        self.offsets.append(offset)
        self.i_specimen += 1

        if self.i_specimen == len(self.specimen_pitches):
            self.report()
        else:
            self.update_plots()
   

    def update_plots(self):
        '''
        Call to update imshow plots.
        '''
        self.reference_image = tifffile.imread(self.reference_images[self.i_reference])
        self.reference_plotter.imshow(self.reference_image, cmap='gray', slider=True)

        self.specimen_image = tifffile.imread(self.specimen_images[self.i_specimen])
        self.specimen_plotter.imshow(self.specimen_image, cmap='gray', slider=True)

    
    def set_manual(self):
        '''
        Let the user specify a manual correction, skipping the rotation process.
        '''
        value = tk.simpledialog.askstring("Manual correction value",
                "The vertical angle when the deep\npseudopupils align with the antenna?", parent=self)
        
        self.offsets = float(value)
        self.report()

    def report(self):
        '''
        Report the results with a pop up window
        '''
        message = 'Correction value {}'.format(np.mean(self.offsets))
        tk.messagebox.showinfo('Zero correction ready', message)
        
        save_antenna_level_correction(self.specimen_name, np.mean(self.offsets))
        
        if self.callback:
            self.callback()

        self.destroy()


class RepetitionSelector(tk.Frame):

    def __init__(self, tk_master, RecordingPlotter, core, update_command):
        '''
        
        update_command      Callable that updates the plots, no input arguments
        '''

        self.update_command = update_command

        tk.Frame.__init__(self, tk_master)
        self.core = core

        self.plotter = RecordingPlotter
            
        self.text = tk.StringVar()
        self.infotext = tk.Label(self, textvariable = self.text)
        self.infotext.grid(row=0, column=0)

        self.all = tk.Button(self, text='Show all', command=lambda: self.move_selection(None))
        self.all.grid(row=0, column=1)

        self.previous = tk.Button(self, text='Previous', command=lambda: self.move_selection(-1))
        self.previous.grid(row=0, column=2)

        self.next = tk.Button(self, text='Next', command=lambda: self.move_selection(1))
        self.next.grid(row=0, column=3)

        self.mark_bad = tk.Button(self, text='Mark bad', command=self.mark_bad)
        self.mark_bad.grid(row=0, column=4)

    def mark_bad(self):
        im_folder = self.core.selected_recording
        
        if self.plotter.i_repeat == None:
            pass
        else:
            self.core.analyser.mark_bad(im_folder, self.plotter.i_repeat)
        

    def move_selection(self, direction):
        '''
        None        sets plotter to show all repeats
        1 or -1     Move to next/previous repetition
        '''

        if direction == None:
            self.plotter.i_repeat = None
        else:
            if self.plotter.i_repeat == None:
                self.plotter.i_repeat = 0
            else:
                self.plotter.i_repeat += direction

            if self.plotter.i_repeat < 0:
                self.plotter.i_repeat = 0
            elif self.plotter.i_repeat >= self.plotter.N_repeats:
                self.plotter.i_repeat = self.plotter.N_repeats -1
        
        self.update_text()
        
        print(self.plotter.i_repeat)

        if self.update_command:
            self.update_command()

    def update_text(self):
        if self.plotter.i_repeat is None:
            isel = None
        else:
            isel = self.plotter.i_repeat + 1
        self.text.set("{}/{}".format(isel, self.plotter.N_repeats))
