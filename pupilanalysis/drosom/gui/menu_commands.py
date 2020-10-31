'''
This module contains the menu bar command classes, that inherit from
tk_steroids' MenuMaker for easy initialization.

In the beginning of the module, there are some needed functions.
'''


import os

import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

from tk_steroids.menumaker import MenuMaker

from pupilanalysis.droso import SpecimenGroups
from pupilanalysis.drosom import linked_data



def ask_string(title, prompt):
    '''
    Asks the user for a string.
    '''
    string = simpledialog.askstring(title, prompt, parent=self.parent)
    return string



def prompt_result(self, string):
    '''
    Shows the result and also sets it to the clipboard
    '''
    self.root.clipboard_clear()
    self.root.clipboard_append(string)

    messagebox.showinfo(title='Result of ', message=string)



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
    top.grid_columnconfigure(0, weight=1)
    top.grid_rowconfigure(1, weight=1)


    if with_rois or with_movements or with_correction:
        notify_string = 'Listing specimens with '
        notify_string += ' and '.join([string for onoff, string in zip([with_rois, with_movements, with_correction],
            ['ROIs', 'movements', 'correction']) if onoff ])
        tk.Label(top, text=notify_string).grid()

    specimens = core.list_specimens(with_rois=with_rois, with_movements=with_movements, with_correction=with_correction) 
    
    if return_manalysers:
        # This is quite wierd what is going on here
        def commandx(specimens, *args, **kwargs):
            manalysers = [core.get_manalyser(specimen) for specimen in specimens]
            return command(manalysers, *args, **kwargs)
    else:
        commandx = command

    selector = TickSelect(top, specimens, commandx, callback_args=parsed_args)

    selector.grid(sticky='NSEW')
    
    tk.Button(selector, text='Close', command=top.destroy).grid(row=1, column=1)




class ModifiedMenuMaker(MenuMaker):
    '''
    Modify the MenuMaker so that at object initialization, we pass
    the common core to all of the command objects.
    '''

    def __init__(self, tk_root, core, *args, **kwargs):
        '''
        core        An instance of the core.py Core Class
        '''
        super().__init__(*args, **kwargs)
        self.core = core
        self.tk_root = tk_root

        self.replacement_dict['__DASH__'] = ' - '
        self.replacement_dict['_'] = ' '



class FileCommands(ModifiedMenuMaker):
    '''
    File menu commands for examine view.
    '''

    def force_order(self):
        '''
        Let's force menu ordering. See the documentation from the
        menu maker.
        '''
        menu = ['set_data_directory',
                'set_hidden_specimens',
                '.',
                'exit']
        return menu


    def set_data_directory(self):
        '''
        Asks user for the data directory.
        '''
        directory = filedialog.askdirectory()
        if not directory:
            return None
    
        self.core.set_data_directory(directory)
        
        self.mainview.update_specimens()

    
    def set_hidden_specimens(self):
        string = self.core.get_hidden()

        newstring = simpledialog.askstring('Set hidden specimens', 'Hidestring (comma separated)',
                initialvalue=string, parent=self)
        
        self.core.set_hidden(newstring)
        self.mainview.update_specimens()


    def exit(self):
        self.tk_root.winfo_toplevel().destroy()




class ImageFolderCommands(ModifiedMenuMaker):
    '''
    Commands for the currently selected image folder.
    '''
    
    def max_of_the_mean_response(self):
        
        result = mean_max_response(self.core.analyser, self.core.selected_recording)
        prompt_result(result)




class SpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for the currentlt selected specimen.
    '''

    def vectormap__DASH__interactive_plot(self):
        self.core.adm_subprocess('current', 'vectormap')


    def vectormap__DASH__rotating_video(self):
        self.core.adm_subprocess('current', 'tk_waiting_window vectormap animation')

    
    def vectormap__DASH__export_npy(self):
        analysername = self.core.analyser.get_specimen_name()
        fn = tk.filedialog.asksaveasfilename(initialfile=analysername, defaultextension='.npy')
        if fn:
            base = fn.rstrip('.npy')
            for eye in ['left', 'right']:
                d3_vectors = self.core.analyser.get_3d_vectors(eye)
                np.save(base+'_'+eye+'.npy', d3_vectors)

    
    def mean_displacement_over_time(self):
        command=lambda: self.core.adm_subprocess('current', 'magtrace')




class ManySpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for all of the specimens in the current data directory.
    Usually involves a checkbox to select the wanted specimens.
    '''
    
    def _batch_measure(self, specimens):
        targets = [self.core.get_manalyser(specimen).measure_both_eyes for specimen in specimens]
        MeasurementWindow(self.parent_menu.winfo_toplevel(), targets, title='Measure movement', callback_on_exit=self.core.update_gui)


    def measure_movements__DASH__list_all(self):

        select_specimens(self.core, self._batch_measure, with_rois=True)


    def measure_movements__DASH__list_only_unmeasured(self):

        select_specimens(self._batch_measure, with_rois=True, with_movements=False)


    def create_specimens_group(self):
        '''
        Group specimens for fast, repeated selection
        '''

        group_name = ask_string('Group name', 'Name the new group')
        
        groups = SpecimenGroups()
        groups.new_group(group_name, *specimens)
        groups.save_groups()

        select_specimens(self.core, _create_group)


    def averaged_vectormap__DASH__interactive_plot(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged'), with_movements=True, with_correction=True)


    
    def averaged_vectormap__DASH__rotating_video(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged vectormap animation'), with_movements=True, with_correction=True) 

        
    def averaged_vectormap__DASH__rotating_video__DASH__set_title(self):
        ask_string('Set title', 'Give video title', lambda title: select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged vectormap animation short_name={}'.format(title)), with_movements=True, with_correction=True)) 
        
        
    def comparision_to_optic_flow__DASH__video(self): 
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged complete_flow_analysis'), with_movements=True, with_correction=True) 
        
    
    def link_ERG_data_from_labbook(self):
        select_specimens(self.core, linked_data.link_erg_labbook, command_args=[lambda: filedialog.askopenfilename(title='Select ERG'), lambda: filedialog.askdirectory(title='Select data folder')], return_manalysers=True )
        




