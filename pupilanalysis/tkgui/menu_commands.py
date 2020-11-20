'''
This module contains the menu bar command classes, that inherit from
tk_steroids' MenuMaker for easy initialization.

In the beginning of the module, there are some needed functions.
'''


import os

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

from tk_steroids.dialogs import TickSelect
from tk_steroids.menumaker import MenuMaker

import pupilanalysis
from pupilanalysis.directories import USER_HOMEDIR
from pupilanalysis.droso import SpecimenGroups
from pupilanalysis.drosom import linked_data
from pupilanalysis.drosom.kinematics import mean_max_response
from pupilanalysis.tkgui import settings
from pupilanalysis.tkgui.run_measurement import MeasurementWindow
from pupilanalysis.tkgui.zero_correct import ZeroCorrect


def ask_string(title, prompt, tk_parent):
    '''
    Asks the user for a string.
    '''
    string = simpledialog.askstring(title, prompt, parent=tk_parent)
    return string



def prompt_result(tk_root, string):
    '''
    Shows the result and also sets it to the clipboard
    '''
    tk_root.clipboard_clear()
    tk_root.clipboard_append(string)

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

        self.replacement_dict['DASH'] = '-'
        self.replacement_dict['_'] = ' '



class FileCommands(ModifiedMenuMaker):
    '''
    File menu commands for examine view.
    '''

    def _force_order(self):
        '''
        Let's force menu ordering. See the documentation from the
        menu maker.
        '''
        menu = ['set_data_directory',
                'settings',
                '.',
                'exit']
        return menu


    def set_data_directory(self):
        '''
        Asks user for the data directory.
        '''
        previousdir = settings.get('last_datadir', default=USER_HOMEDIR)
        
        # Check if the previous data directory stil exists (usb drive for example)
        if not os.path.isdir(previousdir):
            previousdir = USER_HOMEDIR 

        directory = filedialog.askdirectory(
                parent=self.tk_root,
                title='Select directory containing specimens',
                mustexist=True,
                initialdir=previousdir
                )

        if not directory:
            return None
    
        self.core.set_data_directory(directory)
        self.core.update_gui(changed_specimens=True)
        
        settings.set('last_datadir', directory)

    
    def settings(self):
        pass


    def exit(self):
        self.tk_root.winfo_toplevel().destroy()




class ImageFolderCommands(ModifiedMenuMaker):
    '''
    Commands for the currently selected image folder.
    '''
   
    def _force_order(self):
        return ['select_ROIs', 'measure_movement',
                '.',
                'max_of_the_mean_response']

    def max_of_the_mean_response(self):
        
        result = mean_max_response(self.core.analyser, self.core.selected_recording)
        prompt_result(self.tk_root, result)


    def select_ROIs(self):
        self.core.analyser.select_ROIs(callback_on_exit=self.core.update_gui,
                reselect_fns=[self.core.selected_recording], old_markings=True)


    def measure_movement(self):
        '''
        Run Movemeter (cross-correlation) on the selected image folder.
        '''
        func = lambda: self.core.analyser.measure_both_eyes(only_folders=str(self.core.selected_recording))
        
        MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))
 



class SpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for the currentlt selected specimen.
    '''

    def _force_order(self):
        return ['select_ROIs', 'measure_movement', 'zero_correct',
                '.',
                'measure_movement_DASH_in_absolute_coordinates',
                '.',
                'mean_displacement_over_time',
                '.']

    def select_ROIs(self):
        '''
        Select regions of interests (ROIs) for the currently selected specimen.
        '''
            
        # Ask confirmation if ROIs already selected
        if self.core.analyser.are_rois_selected():
            sure = messagebox.askokcancel('Reselect ROIs', 'Are you sure you want to reselect ROIs?')
            if not sure:
                return None
       
        self.core.analyser.select_ROIs(callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))



    def measure_movement(self, absolute_coordinates=False):
        '''
        Run Movemeter (cross-correlation) on the specimen.
        '''
        
        # Ask confirmation if ROIs already selected
        if self.core.analyser.is_measured():
            sure = messagebox.askokcancel('Remeasure movements', 'Are you sure you want to remeasure?')
            if not sure:
                return None
        
        func = lambda: self.core.analyser.measure_both_eyes(absolute_coordinates=absolute_coordinates)
        
        MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))
    

    def measure_movement_DASH_in_absolute_coordinates(self):
        self.measure_movement(absolute_coordinates=True)


    def zero_correct(self):
        '''
        Start antenna level search for the current specimen 
        '''
        
        # Try to close and destroy if any other antenna_level
        # windows are open (by accident)
        try:
            self.correct_window.destroy()
        except:
            # Nothing to destroy
            pass

        #fullpath = os.path.join(self.data_directory, self.current_specimen)
        #self.core.adm_subprocess('current', 'antenna_level', open_terminal=True)
        self.correct_window = tk.Toplevel()
        self.correct_window.title('Zero correction -  {}'.format(self.current_specimen))
        self.correct_window.grid_columnconfigure(0, weight=1)
        self.correct_window.grid_rowconfigure(0, weight=1)

        def callback():
            self.correct_window.destroy()
            self.core.update_gui()

        self.correct_frame = ZeroCorrect(self.correct_window,
                os.path.join(self.core.directory, self.core.current_specimen), 
                'alr_data',
                callback=callback)
        self.correct_frame.grid(sticky='NSEW')



    def vectormap_DASH_interactive_plot(self):
        self.core.adm_subprocess('current', 'vectormap')


    def vectormap_DASH_rotating_video(self):
        self.core.adm_subprocess('current', 'tk_waiting_window vectormap animation')

    
    def vectormap_DASH_export_npy(self):
        analysername = self.core.analyser.get_specimen_name()
        fn = tk.filedialog.asksaveasfilename(initialfile=analysername, defaultextension='.npy')
        if fn:
            base = fn.rstrip('.npy')
            for eye in ['left', 'right']:
                d3_vectors = self.core.analyser.get_3d_vectors(eye)
                np.save(base+'_'+eye+'.npy', d3_vectors)

    
    def mean_displacement_over_time(self):
        self.core.adm_subprocess('current', 'magtrace')




class ManySpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for all of the specimens in the current data directory.
    Usually involves a checkbox to select the wanted specimens.
    '''

    def _force_order(self):
        return ['measure_movements_DASH_list_all', 'measure_movements_DASH_list_only_unmeasured',
                'measure_movements_DASH_in_absolute_coordinates',
                '.',
                'averaged_vectormap_DASH_interactive_plot', 'averaged_vectormap_DASH_rotating_video',
                'averaged_vectormap_DASH_rotating_video_DASH_set_title',
                '.',
                'comparision_to_optic_flow_DASH_video',
                '.',
                'create_specimens_group',
                'link_ERG_data_from_labbook']
    
    def _batch_measure(self, specimens, absolute_coordinates=False):
        
        # Here lambda requires specimen=specimen keyword argument; Otherwise only
        # the last specimen gets analysed N_specimens times
        targets = [lambda specimen=specimen: self.core.get_manalyser(specimen).measure_both_eyes(absolute_coordinates=absolute_coordinates) for specimen in specimens]

        MeasurementWindow(self.parent_menu.winfo_toplevel(), targets, title='Measure movement',
                callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))


    def measure_movements_DASH_list_all(self):

        select_specimens(self.core, self._batch_measure, with_rois=True)


    def measure_movements_DASH_list_only_unmeasured(self):

        select_specimens(self.core, self._batch_measure, with_rois=True, with_movements=False)

    
    def measure_movements_DASH_in_absolute_coordinates(self):
        func = lambda specimens: self._batch_measure(specimens, absolute_coordinates=True)
        select_specimens(self.core, func, with_rois=True, with_movements=False)



    def create_specimens_group(self):
        '''
        Group specimens for fast, repeated selection
        '''

        group_name = ask_string('Group name', 'Name the new group')
       
        def _create_group(specimens):
            groups = SpecimenGroups()
            groups.new_group(group_name, *specimens)
            groups.save_groups()

        select_specimens(self.core, _create_group)


    def averaged_vectormap_DASH_interactive_plot(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged'), with_movements=True, with_correction=True)


    
    def averaged_vectormap_DASH_rotating_video(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged vectormap animation'), with_movements=True, with_correction=True) 

        
    def averaged_vectormap_DASH_rotating_video_DASH_set_title(self):
        ask_string('Set title', 'Give video title', lambda title: select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged vectormap animation short_name={}'.format(title)), with_movements=True, with_correction=True)) 
        
        
    def comparision_to_optic_flow_DASH_video(self): 
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, 'tk_waiting_window averaged complete_flow_analysis'), with_movements=True, with_correction=True) 
        
    
    def link_ERG_data_from_labbook(self):
        select_specimens(self.core, linked_data.link_erg_labbook, command_args=[lambda: filedialog.askopenfilename(title='Select ERG'), lambda: filedialog.askdirectory(title='Select data folder')], return_manalysers=True )
        

class OtherCommands(ModifiedMenuMaker):
    '''
    All kinds of various commands and tools.
    '''

    def about(self):
        message = 'Pupil analysis'
        message += "\nVersion {}".format(pupilanalysis.__version__)
        message += '\n\nGPL-3.0 License'
        tk.messagebox.showinfo(title='About', message=message)

