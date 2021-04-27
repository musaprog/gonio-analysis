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

from tk_steroids.dialogs import TickSelect, popup_tickselect
from tk_steroids.menumaker import MenuMaker
from tk_steroids.datamanager import ListManager
from tk_steroids.elements import Tabs

import pupilanalysis
from pupilanalysis.directories import USER_HOMEDIR, ANALYSES_SAVEDIR
from pupilanalysis.droso import SpecimenGroups
from pupilanalysis.drosom import linked_data
from pupilanalysis.drosom import kinematics
from pupilanalysis.drosom import sinesweep
from pupilanalysis.drosom.reports.left_right import left_right_displacements, lrfiles_summarise
from pupilanalysis.drosom.reports.stats import response_magnitudes
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
                'add_data_directory',
                'settings',
                '.',
                'exit']
        return menu


    def set_data_directory(self, append=False):
        '''
        Asks user for the data directory and sets it active in Core.
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
        
        if append == False:
            self.core.set_data_directory([directory])
        else:
            self.core.set_data_directory(self.core.data_directory + [directory])
        self.core.update_gui(changed_specimens=True)
        
        settings.set('last_datadir', directory)


    def add_data_directory(self):
        '''
        Like set_data_directory, but instead of changing the data directory,
        comdines the entries from previous and the new data directories.
        '''
        self.set_data_directory(append=True)

    
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
        
        result = kinematics.mean_max_response(self.core.analyser, self.core.selected_recording)
        prompt_result(self.tk_root, result)
    

    def latency_by_sigmoidal_fit(self):

        result = kinematics.sigmoidal_fit(self.core.analyser, self.core.selected_recording)[2]
        prompt_result(self.tk_root, str(np.mean(result)))
    

    def select_ROIs(self):
        self.core.analyser.select_ROIs(callback_on_exit=self.core.update_gui,
                reselect_fns=[self.core.selected_recording], old_markings=True)


    def measure_movement(self, absolute_coordinates=False):
        '''
        Run Movemeter (cross-correlation) on the selected image folder.
        '''
        func = lambda: self.core.analyser.measure_both_eyes(only_folders=str(self.core.selected_recording), absolute_coordinates=absolute_coordinates)
        
        MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))


    def measure_movement_DASH_in_absolute_coordinates(self):
        self.measure_movement(absolute_coordinates=True)



class SpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for the currentlt selected specimen.
    '''

    def _force_order(self):
        return ['set_active_analysis', 'set_vector_rotation',
                '.',
                'select_ROIs', 'measure_movement', 'zero_correct',
                '.',
                'measure_movement_DASH_in_absolute_coordinates',
                '.',
                'mean_displacement_over_time',
                '.']


    def set_active_analysis(self):

        name = ask_string('Active analysis', 'Give new or existing analysis name (empty for default)', self.tk_root)
        
        self.core.analyser.active_analysis = name
        self.tk_root.status_active_analysis.config(text='Active analysis: {}'.format(self.core.analyser.active_analysis))
    

    def set_vector_rotation(self):

        rotation = ask_string('Active analysis', 'Give new or existing analysis name (empty for default)', self.tk_root)
        
        if rotation:
            self.core.analyser.vector_rotation = float(rotation)
        else:
            self.core.analyser.vector_rotation = None

        

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
        
        if self.core.analyser.__class__.__name__ == 'MAnalyser':
            MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))
        else:
            # OAnalyser; Threading in MeasurementWindow would cause problems for plotting
            func()
            self.core.update_gui(changed_specimens=True)


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
        self.core.adm_subprocess('current', '--tk_waiting_window vectormap_video')

    
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


    def mean_latency_by_sigmoidal_fit(self):
        results_string = ''
        for image_folder in self.core.analyser.list_imagefolders():
            result = kinematics.sigmoidal_fit(self.core.analyser, image_folder)[2]
            results_string += '{}   {}'.format(image_folder, np.mean(result))
        

        prompt_result(self.tk_root, results_string)


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
                'export_LR_displacement_CSV',
                'export_LR_displacement_CSV_DASH_strong_weak_eye_division',
                'save_kinematics_analysis_CSV',
                'save_sinesweep_analysis_CSV']


    def _batch_measure(self, specimens, absolute_coordinates=False):
        
        # Here lambda requires specimen=specimen keyword argument; Otherwise only
        # the last specimen gets analysed N_specimens times
        targets = [lambda specimen=specimen: self.core.get_manalyser(specimen).measure_both_eyes(absolute_coordinates=absolute_coordinates) for specimen in specimens]
    

        if self.core.analyser_class.__name__ == 'MAnalyser':
            MeasurementWindow(self.parent_menu.winfo_toplevel(), targets, title='Measure movement',
                    callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))
        else:
            # For OAnalyser; Threading in MeasurementWindow causes problems for plotting
            for target in targets:
                target()
            self.core.update_gui(changed_specimens=True)


    def measure_movements_DASH_list_all(self):

        select_specimens(self.core, self._batch_measure, with_rois=True)


    def measure_movements_DASH_list_only_unmeasured(self):

        select_specimens(self.core, self._batch_measure, with_rois=True, with_movements=False)

    
    def measure_movements_DASH_in_absolute_coordinates(self):
        func = lambda specimens: self._batch_measure(specimens, absolute_coordinates=True)
        select_specimens(self.core, func, with_rois=True)



    def averaged_vectormap_DASH_interactive_plot(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average vectormap'), with_movements=True)


    
    def averaged_vectormap_DASH_rotating_video(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average vectormap_video'), with_movements=True) 


    def averaged_vectormap_DASH_rotating_video_multiprocessing(self):
        
        def run_workers(specimens):
            if len(specimens) > 0:
                N_workers = os.cpu_count()
                for i_worker in range(N_workers):
                    if i_worker != 0:
                        additional = '--dont-show'
                    else:
                        additional = ''
                    self.core.adm_subprocess(specimens, '--tk_waiting_window --worker-info {} {} --average vectormap_video'.format(i_worker, N_workers)) 


        select_specimens(self.core, run_workers, with_movements=True) 
        

    def averaged_vectormap_DASH_rotating_video_DASH_set_title(self):
        ask_string('Set title', 'Give video title', lambda title: select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average --short-name {} vectormap_video'.format(title)), with_movements=True)) 
        
        
    def comparision_to_optic_flow_DASH_video(self): 
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average flow_analysis_pitch'), with_movements=True) 
        
    
    
    def export_LR_displacement_CSV(self, strong_weak_division=False):
        '''
        Grouped to left right
        '''
        def callback(specimens):
            group_name = ask_string('Group name', 'Name the selected group of specimens', self.tk_root)
            analysers = self.core.get_manalysers(specimens)
            left_right_displacements(analysers, group_name,
                    strong_weak_division=strong_weak_division) 

        select_specimens(self.core, callback, with_movements=True) 
    

    def export_LR_displacement_CSV_DASH_strong_weak_eye_division(self):
        '''
        Grouped to strong vs weak eye.
        '''
        self.export_LR_displacement_CSV(strong_weak_division=True)


    def save_kinematics_analysis_CSV(self):

        def callback(specimens):
            
            fn = tk.filedialog.asksaveasfilename(title='Save kinematics analysis', initialfile='latencies.csv')
            
            if fn:
                analysers = self.core.get_manalysers(specimens)
                kinematics.save_sigmoidal_fit_CSV(analysers, fn)
 

        select_specimens(self.core, callback, with_movements=True) 


    def save_sinesweep_analysis_CSV(self):
        def callback(specimens):
            analysers = self.core.get_manalysers(specimen)                
            sinesweep.save_sinesweep_analysis_CSV(analysers)
 

        select_specimens(self.core, callback, with_movements=True) 


    def response_magnitude_stats(self):
        def callback(grouped_manalysers):
            response_magnitudes(grouped_manalysers)

        select_specimen_groups(self.core, callback)


    def LR_stasts(self):

        fns = filedialog.askopenfilenames(initialdir=ANALYSES_SAVEDIR)
        
        if fns:
            lrfiles_summarise(fns)



class OtherCommands(ModifiedMenuMaker):
    '''
    All kinds of various commands and tools.
    '''
    
    def _force_order(self):
        return ['manage_specimen_groups',
                'link_ERG_data_from_labbook',
                '.',
                'change_Analyser_DASH_object',
                '.',
                'about']


    def manage_specimen_groups(self):
        '''
        Fixme: This is a little hacked together.
        '''
        
        def _preedit():
            select_specimens(self.core, _postedit)

        def _postedit(specimens):
            self.dm.im2.set_data(specimens)
            self.dm.im2.postchange_callback(self.dm.im2.data)

        def onsave():
            self.groups.groups = self.dm.im1.data
            self.groups.save_groups()


        def oncancel():
            top.destroy()

        self.groups = SpecimenGroups() 
        
        top = tk.Toplevel(self.tk_root)

        self.dm = ListManager(top, start_data=self.groups.groups,
                save_callback=onsave, cancel_callback=oncancel)
        tk.Button(self.dm.im2.buttons, text='Select specimens', command=_preedit).grid()
        self.dm.grid(row=1, column=1, sticky='NSWE')
        top.rowconfigure(1, weight=1)
        top.columnconfigure(1, weight=1)

        top.mainloop()

    
    def link_ERG_data_from_labbook(self):
        select_specimens(self.core, linked_data.link_erg_labbook, command_args=[lambda: filedialog.askopenfilename(title='Select ERG'), lambda: filedialog.askdirectory(title='Select data folder')], return_manalysers=True )
 


    def change_Analyser_DASH_object(self):

        popup_tickselect(self.tk_root,
                [c.__name__ for c in self.core.analyser_classes],
                lambda selections: self.core.set_analyser_class(selections[0]),
                ticked=[self.core.analyser_class],
                single_select=True)


    def about(self):
        message = 'Pupil analysis'
        message += "\nVersion {}".format(pupilanalysis.__version__)
        message += '\n\nGPL-3.0 License'
        tk.messagebox.showinfo(title='About', message=message)

