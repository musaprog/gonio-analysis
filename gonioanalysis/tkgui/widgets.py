'''
Gonio-analysis tkinter GUI widgets.
'''
import os
import numpy as np
import threading

import tkinter as tk
from tkinter import filedialog
import tifffile

from tk_steroids.elements import (Tabs,
        ButtonsFrame,
        TickboxFrame,
        Listbox,
        )
from tk_steroids.matplotlib import CanvasPlotter
from tk_steroids.dialogs import TickSelect
from tk_steroids.routines import inspect_booleans
from tk_steroids.menumaker import MenuMaker

from gonioanalysis.droso import SpecimenGroups
from gonioanalysis.drosom.analysing import MAverager
from gonioanalysis.antenna_level import (
        load_drosom,
        save_antenna_level_correction,
        load_reference_fly,
        )
from gonioanalysis.drosom.plotting.basics import (
        plot_3d_vectormap,
        plot_3d_differencemap,
        )







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
                button_names=['Add', 'Remove', 'Ok'], horizontal=False,
                button_commands=[self.on_add_press, self.on_remove_press, self.on_ok])
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


    def on_add_press(self):
        image_folder = self.imagefolders_listbox.current
        if image_folder:
            sel = self.specimens_listbox.current + self._separator + image_folder
            
            selections = self.selected_listbox.selections + [sel]
            self.selected_listbox.set_selections(selections)

    
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
       
        try:
            alr_data = load_reference_fly(alr_data_path)
        except FileNotFoundError:
            alr_data = {}

        self.reference_pitches, self.reference_images = [[],[]]
        for pitch, fn in sorted(alr_data.items(), key=lambda x: float(x[0])):
            self.reference_pitches.append(pitch)
            self.reference_images.append(fn)

        # Set plotters
        self.specimen_plotter = CanvasPlotter(self, text=specimen_path)
        self.specimen_plotter.grid(row=1, column=0, sticky='NSWE')

        self.reference_plotter = CanvasPlotter(self, text='Reference fly')
        self.reference_plotter.grid(row=1, column=1, sticky='NSEW')
        
        if not alr_data:
            self.reference_plotter.ax.text(0.1, 0.1, (
                    "No reference data created. Options:\n"
                    "  A) Set correction manually\n"
                    "  B) Run scripts/create_alr_data.py"),
                    color='red')

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
        if self.reference_images:
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
        
        if value:
            self.offsets = float(value)
            self.report()


    def report(self):
        '''
        Report the results with a pop up window
        '''
        message = 'Correction value set as {}'.format(np.mean(self.offsets))
        tk.messagebox.showinfo('Zero correction ready', message, parent=self)
        
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



class RotationButtons(tk.Frame):
    '''
    Create buttons to set a matplotlib 3D plot rotations
    
    Attrubutes
    ----------
    axes : list
        Associated matplotlib axes that get rotated
    buttons_frame : object
        Buttons frame object containing the tkinter under the buttons attribute
    rotation_offset : tuple
        (elev, azim) offset in rotations
    callback : callable or None
        Additional callback to be called after changing rotation.
    '''

    def __init__(self, tk_parent, axes, rotations, callback=None,
            label='', hide_none=True, rotation_offset=(0,0)):
        '''
        tk_parent : object
            Tkinter parent object
        axes : list
            List of matplotlib axes
        rotations : list of tuples
            Rotations [(elev, azim), ...]. If any None, keeps the corresponding
            rotation as it is.
        callback : None or callable
            Callback after each rotation update
        hide_none : bool
            When one of the rotations is None, hide the None from
            button text
        rotation_offset : tuple
            Offset in elevation and azitmuth, respectively in degrees
        '''
        tk.Frame.__init__(self, tk_parent)
        
        self.axes = axes
        self.rotation_offset = rotation_offset
        
        if hide_none:
            names = []
            for rotation in rotations:
                if None in rotation:
                    for r in rotation:
                        if r is not None:
                            names.append(r)
                else:
                    names.append(rotation)
        else:
            names = rotations

        commands = [lambda rot=rot: self.set_rotation(*rot) for rot in rotations]
        self.buttons_frame = ButtonsFrame(self, names, commands, label=label)
        self.buttons_frame.grid(row=1, column=2)
       
        self.callback = callback


    def set_rotation(self, elev, azim):
        for ax in self.axes:
            if elev is None:
                uelev = ax.elev
            else:
                uelev = elev + self.rotation_offset[0]

            if azim is None:
                uazim = ax.azim
            else:
                uazim = azim + self.rotation_offset[1]

            ax.view_init(uelev, uazim)

        if callable(self.callback):
            self.callback()



class CompareVectormaps(tk.Frame):
    '''
    Widget to compare two vectormaps interactively to
    each other.

    Attributes
    ----------
    tk_parent : object
        Parent widget
    plot_functions : list
        List of plot functions,
        default [plot_3d_vectormap, plot_3d_vectormap, plot_3d_differencemap]
    canvases : list
        CanvasPlotter objects, canvas2 to show difference
    tickbox_frames : list
        List of tickbox frame objects
    buttons : list
        Corresponding tk.Buttons under the canvases
    analysers : list
        Analyser objects selected by the user (len == 2)
    '''


    def __init__(self, tk_parent, core):
        class FileMenu(MenuMaker):
        
            def __init__(self, main_widget, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.main_widget = main_widget
            
            def save_all_views(self):
                self.main_widget.savefig()

            def close_window(self):
                self.main_widget.tk_parent.destroy()
                
        tk.Frame.__init__(self, tk_parent)
        self.tk_parent = tk_parent
        
        self.core = core

        self.plot_functions = [plot_3d_vectormap, plot_3d_vectormap,
                plot_3d_differencemap]
        self.canvases = []
        self.tickbox_frames = []
        self.buttons = []
        self.analysers = [None, None]

        self.grid_rowconfigure(0, weight=1)

        axes = []
        for i in range(3):
            canvas = CanvasPlotter(self, projection='3d')
            canvas.grid(row=3, column=i, sticky='NSWE')
            self.canvases.append(canvas)

            axes.append(canvas.ax)
            axes[-1].elev = 15
            axes[-1].azim = 60
            
            # Plot settings
            if i in [0,1]:
                cmd = lambda i=i: self.set_vectormap(i_canvas=i)
            else:
                cmd = self.plot_difference
            options, defaults = inspect_booleans(self.plot_functions[i])
            tickboxes = TickboxFrame(self, options, defaults=defaults,
                    callback=cmd)
            tickboxes.grid(row=4, column=i, sticky='NSWE')
            self.tickbox_frames.append(tickboxes)

            # Main buttons
            if i in [0, 1]:
                cmd = lambda i=i: self.select_specimens(i_canvas=i)
                txt = 'Select specimens...'
            else:
                cmd = self.plot_difference
                txt = 'Compare'

            button = tk.Button(self, text=txt, command=cmd)
            button.grid(row=45, column=i)
            self.buttons.append(button)

            self.grid_columnconfigure(i, weight=1)

        
        hors = [-80, -60, -50, -30, -15, 0, 30, 15, 50, 60, 80]
        verts = hors
        
        hors = [(None, hor) for hor in hors]
        verts = [(ver, None) for ver in verts]

        for i, (name, rotations) in enumerate(zip(['Horizontal', 'Vertical'], [hors, verts])):
            self.rotation_buttons = RotationButtons(self, axes, rotations,
                    label=name+' rotation', callback=self._update_canvases,
                    rotation_offset=(0,90))
            self.rotation_buttons.grid(row=i+1, column=0, columnspan=3)

        self.menubar = tk.Menu()
        self.filemenu = FileMenu(self, 'File')
        self.filemenu._connect(self.menubar)
        self.winfo_toplevel().config(menu=self.menubar)


    def _update_canvases(self):
        for i in range(3):
            ax = self.canvases[i].ax
            if ax.dist != 8.5:
                ax.dist = 8.5
 
            self.canvases[i].update()

    def select_specimens(self, i_canvas):
        select_specimens(self.core, self.set_vectormap,
                command_args=[i_canvas], return_manalysers=True,
                with_movements=True)


    def set_vectormap(self, manalysers=None, i_canvas=None):
        import time 
        start_time = time.time()

        canvas = self.canvases[i_canvas]
        ax = canvas.ax
        
        if manalysers is None:
            analyser = self.analysers[i_canvas]
            if analyser is None:
                return None
        else:
            if len(manalysers) > 1:
                analyser = MAverager(manalysers)
            else:
                analyser = manalysers[0]
        
        azim, elev = (ax.azim, ax.elev)
        ax.clear()
        
       
        kwargs = self.tickbox_frames[i_canvas].states
        
        plot_3d_vectormap(analyser, ax=ax, azim=azim, elev=elev,
                mutation_scale=6, scale_length=1.2, **kwargs)
        
        if ax.dist != 8.5:
            ax.dist = 8.5
 
        canvas.update()

        self.analysers[i_canvas] = analyser

        print('took {} seconds'.format(time.time()-start_time))


    def plot_difference(self):
        
        if any([an is None for an in self.analysers]):
            return None
        
        kwargs = self.tickbox_frames[-1].states
        
        ax = self.canvases[-1].ax
        ax.clear()
        plot_3d_differencemap(*self.analysers[0:2], ax=ax, **kwargs)
        
        if ax.dist != 8.5:
            ax.dist = 8.5

        self.canvases[-1].update()

    
    def savefig(self, i_canvas=None, fn=None):
        '''
        Save current images visible on the canvases

        i_canvas : int or None
            If none, save all views by inserting index of the canvas
            to the end of the saved filename.
        fn : string or None
            If not given, save name is asked from the user.
        '''
        
        if i_canvas is None:
            iterate = range(len(self.canvases))
        elif isinstance(i_canvas, int) and i_canvas:
            iterate = [i_canvas]
        else:
            raise ValueError('wrong type for i_canvas: {}'.format(i_canvas))

        if fn is None:
            
            if i_canvas == 'all':
                text = 'Select common save name for the views'
            else:
                text = 'Save image on a view'

            fn = filedialog.asksaveasfilename(title=text)
            
            if not '.' in os.path.basename(fn):
                fn = fn + '.png'

            if fn:
                for i_canvas in iterate:
                    efn = '.'.join(fn.split('.')[:-1]+[str(i_canvas)]+fn.split('.')[-1:])
                    self.canvases[i_canvas].figure.savefig(efn, dpi=600)
