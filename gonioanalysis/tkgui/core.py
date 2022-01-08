
import os
import subprocess
import sys
import platform

from gonioanalysis.droso import SpecimenGroups
from gonioanalysis.directories import CODE_ROOTDIR
from gonioanalysis.drosom.analysing import MAnalyser
from gonioanalysis.drosom.orientation_analysis import OAnalyser
from gonioanalysis.drosom.transmittance_analysis import TAnalyser
from gonioanalysis.directories import ANALYSES_SAVEDIR


class Core:
    '''
    Tkinter independent functions, reusable for other GUI implementations.
    
    Attributes
    ----------
    data_directory : list of strings
        Current data directories
    current_specimen : string
        Name of the current specimen
    analyser : object
        MAnalyser (or OAnalsyer) object of the current specimen
    selected_recording : string
        Selected recording name (image_folder)
    analyser_class : class
        Class of the new analysers to create (MAnalyser or OAnalyser)
    analyser_classes: list of classes
        List of available analyser classes for reference
    active_analysis : string or None
        Name of the active analysis
    '''

    def __init__(self):
        
        self.data_directory = []
        self.current_specimen = None
        self.analyser = None
        self.selected_recording = None
        
        self.analyser_class = MAnalyser
        self.analyser_classes = [MAnalyser, OAnalyser, TAnalyser]
        
        self.active_analysis = None

        self._folders = {}

        self.groups = SpecimenGroups()


    def set_data_directory(self, data_directory):
        '''
        Update Core's knowledge about the currently selected data_directory.

        Arguments
        ---------
        data_directory : list of strings
            List of paths to the data
        '''
        self.data_directory = data_directory
        
        self._folders = {}
        for data_directory in self.data_directory:
            self._folders[data_directory] = os.listdir(data_directory)


    def set_current_specimen(self, specimen_name):
        '''
        Update Core's knowledge about the currently selected specimen.
        '''
        self.current_specimen = specimen_name
        self.analyser = self.get_manalyser(specimen_name)

    
    def set_selected_recording(self, selected_recording):
        self.selected_recording = selected_recording
    

    def set_analyser_class(self, class_name):
        index = [i for i, ac in enumerate(self.analyser_classes) if ac.__name__ == class_name]
        self.analyser_class = self.analyser_classes[index[0]]
        
        if self.data_directory:
            self.update_gui(changed_specimens=True)


    def list_specimens(self, with_rois=None, with_movements=None, with_correction=None):
        '''
        List specimens in the data directory. May contain bad folders also (no check for contents)

        With the following keyword arguments one select only the specimens fulfilling the conditions
        by setting the keyword argument either to True (has to fulfill condition), False (negative)
        or to None (the condition is not considered)

            with_rois           Specimens with ROIs selected
            with_movements      Specimens with movements measured
            with_correction     Specimens with antenna_level (zero level) correction
        
        For example, if you want the specimens with movements but without antenna_level corrections
        and you won't care about ROIs, set
            with_rois=None, with_movements=True, with_correction=False

        '''
        specimens = []
        for data_directory in self.data_directory:
            specimens.extend( [fn for fn in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, fn))] )
        
        if with_rois is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).are_rois_selected() == with_rois]
        
        if with_movements is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).is_measured() == with_movements]
        
        if with_correction is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).get_antenna_level_correction() is not False]

        return sorted(specimens)
    

    def get_specimen_fullpath(self, specimen_name=None):
        '''
        Returns the full path of a specimen (datadir + specimen_patch)
        
        Arguments
        ---------
        specimen_namse : string or None
            If None use self.current_specimen
        '''
        if specimen_name is None:
            specimen_name = self.current_specimen

        for directory in self.data_directory:
            if specimen_name in self._folders[directory]:
                return os.path.join(directory, specimen_name)

        raise ValueError("no specimen with name {}".format(specimen_name))


    def _configure_analyser(self, analyser):
        if self.active_analysis:
            analyser.active_analysis = self.active_analysis
        return analyser


    def get_manalyser(self, specimen_name, **kwargs):
        '''
        Gets manalyser for the specimen specified by the given name.
        '''
        for directory in self.data_directory:
            if specimen_name in self._folders[directory]:
                break

        analyser = self.analyser_class(directory, specimen_name, **kwargs)

        return self._configure_analyser(analyser)
    

    def get_manalysers(self, specimen_names, **kwargs):
        '''
        Like get_manalyser but returns a list of analyser objects and also
        checks for specimen groups if a specimen cannot be found.
        '''
        analysers = []
        for name in specimen_names:
            try:
                ans = [self.get_manalyser(name, **kwargs)]
            except FileNotFoundError:
                ans = [self.get_manalyser(n, **kwargs) for n in self.groups.groups.get(name, [])]
                
                # Try again and load
                if ans is []:
                    self.groups.load_groups()
                    ans = [self.get_manalyser(n, **kwargs) for n in self.groups.groups.get(name, [])]
            
            if ans is []:
                raise FileNotFoundError('Cannot find specimen {}'.format(name))
            
            for an in ans:
                analysers.append( self._configure_analyser(an) )

        return analysers


    def adm_subprocess(self, specimens, terminal_args, open_terminal=False):
        '''
        Invokes drosom/terminal.py
        
        Arguments
        ---------
        specimens : list of string
            List of specimen names or 'current'
        terminal_args : string
            Agruments passed to the plotter
        open_terminal : bool
            If true open in a cmd window (on Windows) or lxterm (on Linux)
        '''
        
        # 1) Find python executable and wrap the filename by quation marks if spaces present
        python = sys.executable
        if ' ' in python:
            python = '"' + python + '"'

       
        # 2) Find the full path to the adm Python file in the gonio root
        pyfile = os.path.join(CODE_ROOTDIR, 'drosom/terminal.py')
        
        # Check for spaces in the filename. If there are spaces in the filename,
        # we have to encapsulate the filename by quation marks
        if ' ' in pyfile:
            pyfile = '"' + pyfile + '"'

        # 3) Get the specimen directoires (full paths separated by space)
        if specimens == 'current':
            specimen_names = self.analyser.folder
        else:
            specimen_names = ':'.join(specimens)
            if not ':' in specimen_names:
                specimen_names += ':'
        
        if self.active_analysis not in ['default', '', None]:
            terminal_args += ' --active-analysis '+ self.active_analysis

        arguments = '-D "{}" -S "{}" {}'.format(' '.join(self.data_directory), specimen_names, terminal_args)
        
        if self.analyser_class is MAnalyser:
            pass
        elif self.analyser_class is OAnalyser:
            arguments = '--type orientation ' + arguments
        elif self.analyser_class is TAnalyser:
            arguments = '--type transmittance ' + arguments
        else:
            raise NotImplementedError

        command = '{} {} {} &'.format(python, pyfile, arguments)
        
        if open_terminal:
            if platform.system() == 'Linux':
                command = 'lxterm -e ' + command
            elif platform.system() == 'Windows':
                command = 'start /wait ' + command
            else:
                raise OSError('Operating system not supported by gonio?')

        print(command)
        
        subprocess.run(command, shell=True)


    def update_gui(self, changed_specimens=False):
        raise ValueError("GUI should overload update_gui method in Core core.py")

