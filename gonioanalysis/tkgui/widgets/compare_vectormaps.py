'''
Gonio-analysis tkinter GUI widgets.
'''
import os
import numpy as np

import tkinter as tk
from tkinter import filedialog

from tk_steroids.elements import (
        ButtonsFrame,
        TickboxFrame,
        )
from tk_steroids.matplotlib import CanvasPlotter
from tk_steroids.routines import inspect_booleans
from tk_steroids.menumaker import MenuMaker

from gonioanalysis.drosom.analysing import MAverager
from gonioanalysis.drosom.plotting.basics import (
        plot_3d_vectormap,
        plot_3d_differencemap,
        )



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
            #if ax.dist != 8.5:
            #    ax.dist = 8.5
 
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
       
        #if ax.dist != 8.5:
        #    ax.dist = 8.5
 
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
        
        #if ax.dist != 8.5:
        #    ax.dist = 8.5

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
