
import os

import tkinter as tk
import tkinter.filedialog as filedialog
from tk_steroids.elements import DropdownList

from .common import ImagefolderMultisel

from gonioanalysis.drosom.reports.left_right import left_right_displacements, lrfiles_summarise
from gonioanalysis.drosom.reports.stats import response_magnitudes
import gonioanalysis.drosom.reports as reports


class ExportWizard(tk.Frame):
    '''Export data from selected specimens and image folders.
    '''
    def __init__(self, tk_parent, core):
        tk.Frame.__init__(self, tk_parent)
        self.tk_parent = tk_parent
        self.core = core
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        self.multisel = ImagefolderMultisel(self, self.core, self.on_export_click)
        self.multisel.grid(row=3, column=0, columnspan=3, sticky='NSWE')
        
        self._export_types = reports.export_docstrings.copy()

        # Export type
        tk.Label(self, text='Export type').grid(row=1, column=0)
        self.export_selection = DropdownList(
                self,
                list(self._export_types.keys()),
                fancynames=list(self._export_types.values()))
        self.export_selection.grid(row=2, column=0, sticky='WE')

        # Save location and name
        tk.Label(self, text='Save folder').grid(row=1, column=1)
        self._folder = str(os.getcwd())
        self.folder = tk.Button(self, text=self._folder, command=self.set_folder)
        self.folder.grid(row=2, column=1, sticky='WE')
        
        #self._name = 'group_name'
        tk.Label(self, text='File name').grid(row=1, column=2)
        self.name = tk.Entry(self)
        self.name.grid(row=2, column=2, sticky='WE')


    def set_folder(self):
        directory = filedialog.askdirectory(
                parent=self.tk_parent,
                title='Select save folder for the export',
                mustexist=True,
                initialdir=self._folder)

        if directory:
            self._folder = directory
            self.folder.config(text=directory)

    
    def on_export_click(self, wanted_imagefolders):
        sel = self.export_selection.ticked[0]

        analysers = self.core.get_manalysers(list(wanted_imagefolders.keys()))

        folder = self._folder
        group_name = str(self.name.get())
        
        if sel in reports.export_functions: 
            reports.export_functions[sel](
                    analysers, group_name,
                    wanted_imagefolders=wanted_imagefolders,
                    savedir=folder
                    )
        elif sel == 'Displacement probability TIFF':
            specimens = [';'.join([specimen, *image_folders]) for specimen, image_folders in wanted_imagefolders.items()]
            self.core.adm_subprocess(specimens, '-A magnitude_probability')
        elif sel == 'XY trajectory plot':
            specimens = [';'.join([specimen, *image_folders]) for specimen, image_folders in wanted_imagefolders.items()]
            self.core.adm_subprocess(specimens, '-A xy_trajectory')
        else:
            raise ValueError('Invalid export type selection')

        # Destroy the popup window 
        self.tk_parent.destroy()
