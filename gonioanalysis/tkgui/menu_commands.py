'''
This module contains the menu bar command classes, that inherit from
tk_steroids' MenuMaker for easy initialization.

In the beginning of the module, there are some needed functions.
'''


import os
import subprocess

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

from tk_steroids.elements import TickboxFrame
from tk_steroids.dialogs import popup_tickselect, popup
from tk_steroids.menumaker import MenuMaker
from tk_steroids.datamanager import ListManager

import gonioanalysis
from gonioanalysis.directories import USER_HOMEDIR, ANALYSES_SAVEDIR
from gonioanalysis.droso import SpecimenGroups
from gonioanalysis.drosom import linked_data
from gonioanalysis.drosom import kinematics
from gonioanalysis.drosom import sinesweep
from gonioanalysis.drosom import export
from gonioanalysis.drosom.reports.left_right import left_right_displacements, lrfiles_summarise
from gonioanalysis.drosom.reports.stats import response_magnitudes
from gonioanalysis.tkgui import settings
from gonioanalysis.tkgui.run_measurement import MeasurementWindow
from gonioanalysis.tkgui.widgets import (
        select_specimens,
        select_specimen_groups,
        ZeroCorrect,
        CompareVectormaps,
        ImagefolderMultisel,
        ExportWizard,
        )




def ask_string(title, prompt, tk_parent):
    '''
    Asks the user for a string.
    '''
    string = simpledialog.askstring(title, prompt, parent=tk_parent)
    return string



def prompt_result(tk_root, string, title='Message'):
    '''
    Shows the result and also sets it to the clipboard
    '''
    tk_root.clipboard_clear()
    tk_root.clipboard_append(string)

    messagebox.showinfo(
            title=title,
            message=f'Following has been copied to your clipboard:\n\n{string}')



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
    
    def _message(self, message, **kwargs):
        if message == 'nospecimen':
            message = 'Select a specimen first'

        prompt_result(self.tk_root, message, **kwargs)
    
    def _ask_string(self, message, title='Input text'):
        return ask_string(title, message, self.tk_root)


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
                '.',
                'create_virtual_analyser',
                '.',
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


    def create_virtual_analyser(self):
        
        def create_virt(specimens):
            self.core.create_virtual_analyser('VirtTest', specimens)
    
        select_specimens(self.core, create_virt, with_movements=True)

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
                'measure_movement_DASH_in_absolute_coordinates',
                '.',
                'max_of_the_mean_response',
                'half_rise_time',
                'half_rise_time_DASH_fit_to_mean',
                'latency',
                'latency_DASH_fit_to_mean',
                '.',
                'start_position_analysis',
                '.',
                'open_in_ImageJ']

    def max_of_the_mean_response(self):
        
        result = kinematics.mean_max_response(self.core.analyser, self.core.selected_recording)
        prompt_result(self.tk_root, result, 'Max of the mean (pixels)')
    

    def half_rise_time(self, fit_to_mean=False):
        result = kinematics.sigmoidal_fit(
                self.core.analyser, self.core.selected_recording,
                fit_to_mean=fit_to_mean)[2]
        prompt_result(self.tk_root, str(np.mean(result)), 'Half-rise time (s)')
    

    def half_rise_time_DASH_fit_to_mean(self):
        self.half_rise_time(fit_to_mean=True)

    def latency(self, fit_to_mean=False):
        result = kinematics.latency(
                self.core.analyser, self.core.selected_recording,
                fit_to_mean=fit_to_mean)
        prompt_result(self.tk_root, str(np.mean(result)), 'Latency (s)')

    def latency_DASH_fit_to_mean(self):
        self.latency(True)

    def select_ROIs(self):
        self.core.analyser.select_ROIs(callback_on_exit=self.core.update_gui,
                reselect_fns=[self.core.selected_recording], old_markings=True)


    def measure_movement(self, absolute_coordinates=False):
        '''
        Run Movemeter (cross-correlation) on the selected image folder.
        '''
        func = lambda stop: self.core.analyser.measure_both_eyes(only_folders=str(self.core.selected_recording), absolute_coordinates=absolute_coordinates, stop_event=stop)
        
        MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))


    def measure_movement_DASH_in_absolute_coordinates(self):
        self.measure_movement(absolute_coordinates=True)
    

    def open_in_ImageJ(self):
        '''Opens the currently selected image folder in ImageJ
        '''
        folder = self.core.selected_recording
        fns = self.core.analyser.list_images(folder, absolute_path=True)
        subprocess.Popen(['imagej', *fns,])
    

    def start_position_analysis(self):
        self.core.adm_subprocess(
                'current',
                f'-A startpos --analysis-options "image_folder={self.core.selected_recording}"')



class SpecimenCommands(ModifiedMenuMaker):
    '''
    Commands for the currentlt selected specimen.
    '''

    def _force_order(self):
        return ['set_active_analysis',
                'set_vector_rotation_offset',
                'set_vertical_zero_rotation',
                'set_yaw_rotation',
                '.',
                'select_ROIs',
                'measure_movement',
                'measure_movement_DASH_in_absolute_coordinates',
                '.',
                'mean_displacement_over_time',
                'latency_by_sigmoidal_fit',
                '.',
                'vectormap_DASH_interactive_plot',
                'vectormap_DASH_rotating_video',
                'vectormap_DASH_export',
                '.',
                'start_position_analysis',
                '.']


    def set_active_analysis(self):

        name = ask_string('Active analysis', 'Give new or existing analysis name (empty for default)', self.tk_root)
        
        self.core.active_analysis = name
        if self.core.analyser:
            self.core.analyser.active_analysis = name
        self.tk_root.status_active_analysis.config(text='Active analysis: {}'.format(self.core.active_analysis))
        
        self.core.update_gui(changed_specimens=True) 


    def set_vector_rotation_offset(self):

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
        if not self.core.current_specimen:
            self._message('nospecimen')
            return None

        # Ask confirmation if ROIs already selected
        if self.core.analyser.is_measured():
            sure = messagebox.askokcancel('Remeasure movements', 'Are you sure you want to remeasure?')
            if not sure:
                return None
        
        func = lambda stop: self.core.analyser.measure_both_eyes(absolute_coordinates=absolute_coordinates, stop_event=stop)
        
        if self.core.analyser.__class__.__name__ != 'OAnalyser':
            MeasurementWindow(self.tk_root, [func], title='Measure movement', callback_on_exit=lambda: self.core.update_gui(changed_specimens=True))
        else:
            # OAnalyser; Threading in MeasurementWindow would cause problems for plotting
            func(stop=None)
            self.core.update_gui(changed_specimens=True)


    def measure_movement_DASH_in_absolute_coordinates(self):
        self.measure_movement(absolute_coordinates=True)


    def set_vertical_zero_rotation(self):
        '''
        Start antenna level search for the current specimen (zero correction)
        '''
        
        # Try to close and destroy if any other antenna_level
        # windows are open (by accident)
        try:
            self.correct_window.destroy()
        except:
            # Nothing to destroy
            pass
        
        if not self.core.current_specimen:
            self._message("nospecimen")
        else:
            
            self.correct_window = tk.Toplevel()
            self.correct_window.title('Zero correction -  {}'.format(self.core.current_specimen))
            self.correct_window.grid_columnconfigure(0, weight=1)
            self.correct_window.grid_rowconfigure(0, weight=1)

            def callback():
                self.correct_window.destroy()
                self.core.update_gui()

            self.correct_frame = ZeroCorrect(self.correct_window,
                    self.core.get_specimen_fullpath(),
                    'alr_data',
                    callback=callback)
            self.correct_frame.grid(sticky='NSEW')


    def set_yaw_rotation(self):
        # Try to close and destroy if any other antenna_level
        # windows are open (by accident)
        try:
            self.yaw_window.destroy()
        except:
            # Nothing to destroy
            pass
        
        if not self.core.current_specimen:
            self._message("nospecimen")
        else:
            
            self.yaw_window = tk.Toplevel()
            self.yaw_window.title(f'Yaw rotation set -  {self.core.current_specimen}')
            self.yaw_window.grid_columnconfigure(0, weight=1)
            self.yaw_window.grid_rowconfigure(0, weight=1)

            def on_yaw():
                
                yaw = self.yaw_frame.ticked[0]
                self.core.analyser.attributes['yaw'] = int(yaw)
                self.core.analyser.save_attributes()

                self.yaw_window.destroy()
                self.core.update_gui()

            yaws = ['-90','0','90']
            yaw = str(self.core.analyser.attributes['yaw'])
            defaults = [False,False,False]
            defaults[yaws.index(yaw)] = True

            self.yaw_frame = TickboxFrame(
                    self.yaw_window, yaws, defaults=defaults,
                    single_select=True)
            self.yaw_frame.grid(
                    row=0, column=0, columnspan=2,
                    sticky='NSEW')
            
            ok = tk.Button(self.yaw_window, text='Ok', command=on_yaw)
            ok.grid(row=1, column=1)
            
            cancel = tk.Button(self.yaw_window, text='Cancel',
                               command=self.yaw_window.destroy)
            cancel.grid(row=1, column=0)

    def vectormap_DASH_interactive_plot(self):
        self.core.adm_subprocess('current', '-A vectormap')


    def vectormap_DASH_rotating_video(self):
        self.core.adm_subprocess('current', '--tk_waiting_window -A vectormap_video')

    
    def vectormap_DASH_export(self):
        analysername = self.core.analyser.get_specimen_name()
        fn = tk.filedialog.asksaveasfilename(initialfile=analysername, defaultextension='.npy', filetypes=export.FILETYPES)
        if fn:
            self.core.adm_subprocess('current', f'-A export_vectormap --output "{fn}"')
            
    
    def mean_displacement_over_time(self):
        self.core.adm_subprocess('current', '-A magtrace --analysis-options mean_repeats=True milliseconds=True mean_imagefolders=True')


    def latency_by_sigmoidal_fit(self):
        results_string = ''
        for image_folder in self.core.analyser.list_imagefolders():
            result = kinematics.sigmoidal_fit(self.core.analyser, image_folder)[2]
            if result is not None:
                result = np.mean(result)
            results_string += '{}   {}\n'.format(image_folder, result)
        

        prompt_result(self.tk_root, results_string, 'Mean latency (s)')

    def start_position_analysis(self):
        self.core.adm_subprocess('current', '-A startpos')


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
                'averaged_vectormap_DASH_export',
                '.',
                'compare_vectormaps',
                '.',
                'comparision_to_optic_flow_DASH_video',
                '.',
                'export_LR_displacement_CSV',
                'export_LR_displacement_CSV_DASH_strong_weak_eye_division',
                'save_kinematics_analysis_CSV',
                'save_kinematics_analysis_CSV_DASH_fit_to_mean',
                'save_sinesweep_analysis_CSV',
                '.',
                'export_wizard',
                '.',
                ]


    def _batch_measure(self, specimens, absolute_coordinates=False):
        
        # Here lambda requires specimen=specimen keyword argument; Otherwise only
        # the last specimen gets analysed N_specimens times
        targets = [lambda stop, specimen=specimen: self.core.get_manalyser(specimen).measure_both_eyes(absolute_coordinates=absolute_coordinates, stop_event=stop) for specimen in specimens]
    

        if self.core.analyser_class.__name__ != 'OAnalyser':
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
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average -A vectormap'), with_movements=True)


    
    def averaged_vectormap_DASH_rotating_video(self):
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average -A vectormap_video'), with_movements=True) 


    def averaged_vectormap_DASH_rotating_video_multiprocessing(self):
        
        def run_workers(specimens):
            if len(specimens) > 0:
                N_workers = os.cpu_count()
                for i_worker in range(N_workers):
                    if i_worker != 0:
                        additional = '--dont-show'
                    else:
                        additional = ''
                    self.core.adm_subprocess(specimens, '--tk_waiting_window --worker-info {} {} --average -A vectormap_video'.format(i_worker, N_workers)) 


        select_specimens(self.core, run_workers, with_movements=True) 
        

    def averaged_vectormap_DASH_rotating_video_DASH_set_title(self):
        ask_string('Set title', 'Give video title', lambda title: select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average --short-name {} -A vectormap_video'.format(title)), with_movements=True)) 
 

    def averaged_vectormap_DASH_export(self):
        def target(specimens):
            fn = tk.filedialog.asksaveasfilename(initialfile='average', defaultextension='.npy')
            if fn:
                self.core.adm_subprocess(specimens, f'--average -A export_vectormap --output {fn}')
        select_specimens(self.core, target)
        


    def compare_vectormaps(self):
        popup(self.tk_root, CompareVectormaps, args=[self.core],
                title='Vectormap comparison')
       

    def comparision_to_optic_flow_DASH_video(self): 
        select_specimens(self.core, lambda specimens: self.core.adm_subprocess(specimens, '--tk_waiting_window --average -A flow_analysis_pitch'), with_movements=True) 
        
    
    
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


    def save_kinematics_analysis_CSV(self, fit_to_mean=False):

        def callback(specimens):
            
            fn = tk.filedialog.asksaveasfilename(title='Save kinematics analysis', initialfile='kinematics.csv')
            
            if fn:
                analysers = self.core.get_manalysers(specimens)
                kinematics.save_sigmoidal_fit_CSV(analysers, fn, fit_to_mean=fit_to_mean)
 
        select_specimens(self.core, callback, with_movements=True) 
    

    def save_kinematics_analysis_CSV_DASH_fit_to_mean(self, ):
        self.save_kinematics_analysis_CSV(fit_to_mean=True)


    def save_sinesweep_analysis_CSV(self):
        def callback(specimens):
            analysers = self.core.get_manalysers(specimens)
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
    
    def LR_kinematics(self):

        fns = filedialog.askopenfilenames(initialdir=ANALYSES_SAVEDIR)
        
        if fns:
            lrfiles_summarise(fns, point_type='kinematics')


    def export_wizard(self):
        top, wizard = popup(self.tk_root, ExportWizard,
                            args=[self.core], title='Export Wizard')
        
   

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
        tk.Button(self.dm.im2.buttons, text='Select specimens', command=_preedit).grid(row=3, column=1)
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
        message = 'Gonio analysis'
        message += "\nVersion {}".format(gonioanalysis.__version__)
        message += '\n\nGPL-3.0 License'
        tk.messagebox.showinfo(title='About', message=message)

