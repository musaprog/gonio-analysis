'''Simple tkinter widgets usable by larger widget assemblies.
'''

import os
import numpy as np
import threading

import tkinter as tk
from tkinter import filedialog

from tk_steroids.elements import (Tabs,
        ButtonsFrame,
        Listbox,
        )
from tk_steroids.dialogs import TickSelect
from gonioanalysis.droso import SpecimenGroups



def select_specimens(core, command, with_rois=None, with_movements=None, with_correction=None,
        command_args=[], execute_callable_args=True, breaking_args=[()],
        return_manalysers=False):
    '''
    Opens a specimen selection window and after ok runs command using
    selected specimens list as the only input argument.

    command                 Command to close after fly selection
    with_rois               List specimens with ROIs selected if True
    with_movements          List specimens with movements measured if True
    command_args            A list of arguments passed to the command
    execute_callable_args   Callable command_args will get executed and return
                                value is used instead
    breaking_args           If command_args callable return value in this list,
                                halt the command
    return_manalysers       Instead of passing the list of the specimen names as the first
                            argument to the command, already construct MAnalyser objects and pass those
    '''
    parsed_args = []
    for arg in command_args:
        if execute_callable_args and callable(arg):
            result = arg()
            if result in breaking_args:
                # Halting
                return None
            parsed_args.append(result)
        else:
            parsed_args.append(arg)


    top = tk.Toplevel()
    top.title('Select specimens')
    top.grid_columnconfigure(1, weight=1)
    top.grid_rowconfigure(1, weight=1)


    if with_rois or with_movements or with_correction:
        notify_string = 'Listing specimens with '
        notify_string += ' and '.join([string for onoff, string in zip([with_rois, with_movements, with_correction],
            ['ROIs', 'movements', 'correction']) if onoff ])
        tk.Label(top, text=notify_string).grid(row=0, column=1)

    specimens = core.list_specimens(with_rois=with_rois, with_movements=with_movements, with_correction=with_correction) 
    
    groups = list(SpecimenGroups().groups.keys())

    if return_manalysers:
        # This is quite wierd what is going on here
        def commandx(specimens, *args, **kwargs):
            manalysers = core.get_manalysers(specimens)
            return command(manalysers, *args, **kwargs)
    else:
        commandx = command

    tabs = Tabs(top, ['Specimens', 'Groups'])

    for tab, selectable in zip(tabs.tabs, [specimens, groups]):
        selector = TickSelect(tab, selectable, commandx, callback_args=parsed_args)
        selector.grid(sticky='NSEW', row=1, column=1)
    
        tk.Button(selector, text='Close', command=top.destroy).grid(row=2, column=1)
    
    tabs.grid(row=1, column=1,sticky='NSEW')


def select_specimen_groups(core, command):
    '''
    command gets the following dictionary
        {'group1_name': [manalyser1_object, ...], ...}
    '''
    top = tk.Toplevel()
    top.title('Select specimen groups')
    top.grid_columnconfigure(0, weight=1)
    top.grid_rowconfigure(1, weight=1)

    
    gm = SpecimenGroups()
    gm.load_groups()

    def commandx(group_names):
        grouped = {}
        for group_name in group_names:
            print(gm.groups[group_name])
            manalysers = [core.get_manalyser(specimen) for specimen in gm.groups[group_name]]
            grouped[group_name] = manalysers
        command(grouped)

    selector = TickSelect(top, list(gm.groups.keys()), commandx)
    selector.grid(sticky='NSEW')

    tk.Button(selector, text='Close', command=top.destroy).grid(row=2, column=1)



class ImagefolderMultisel(tk.Frame):
    '''
    Widget to select image folders from the specimens
    
    Attributes
    ----------
    core
    specimens_listbox
    imagefolders_listbox
    buttons_frame
    '''
    
    def __init__(self, tk_parent, core, callback, **kwargs):
        '''
        *kwargs to core.list_specimens
        '''
        tk.Frame.__init__(self, tk_parent)
        
        self.tk_parent = tk_parent
        self.core = core
        self.callback = callback
        self._separator = ';'

        specimens = core.list_specimens(**kwargs) 
        self.specimens_listbox = Listbox(self, specimens, self.on_specimen_selection)
        self.specimens_listbox.grid(row=0, column=0, sticky='NSWE')
        
        self.imagefolders_listbox = Listbox(self, [''], None)
        self.imagefolders_listbox.grid(row=0, column=1, sticky='NSWE')

        self.buttons_frame = ButtonsFrame(self,
                button_names=['Add', 'Add all', 'Remove', 'Ok'], horizontal=False,
                button_commands=[self.on_add_press, self.on_add_all, self.on_remove_press, self.on_ok])
        self.buttons_frame.grid(row=0, column=2)

        self.selected_listbox = Listbox(self, [], None)
        self.selected_listbox.grid(row=0, column=3, sticky='NSWE')
        
        for i in [0, 1, 3]:
            self.grid_columnconfigure(i, weight=1)
        self.grid_rowconfigure(0, weight=1)
    
    def on_specimen_selection(self, name):
        analyser = self.core.get_manalyser(name)
        image_folders = analyser.list_imagefolders()
        self.imagefolders_listbox.set_selections(image_folders)


    def on_add_press(self, image_folder=None):
        if image_folder is None:
            image_folder = self.imagefolders_listbox.current

        if image_folder:
            sel = self.specimens_listbox.current + self._separator + image_folder
            
            selections = self.selected_listbox.selections + [sel]
            self.selected_listbox.set_selections(selections)
    
    def on_add_all(self):
        for image_folder in self.imagefolders_listbox.selections:
            self.on_add_press(image_folder)

    def on_remove_press(self):
        to_remove = self.selected_listbox.current
        if to_remove:
            selections = self.selected_listbox.selections
            selections.remove(to_remove)
            
            self.selected_listbox.set_selections(selections)


    def on_ok(self):
        image_folders = {}
        for z in self.selected_listbox.selections:
            s, i = z.split(self._separator)
            if s not in image_folders:
                image_folders[s] = []
            image_folders[s].append(i)

        self.callback(image_folders)
        self.tk_parent.destroy()




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



