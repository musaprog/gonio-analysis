import os
import numpy as np
import tkinter as tk
import tifffile


from gonioanalysis.antenna_level import (
        load_drosom,
        save_antenna_level_correction,
        load_reference_fly
        )
from tk_steroids.matplotlib import CanvasPlotter

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




