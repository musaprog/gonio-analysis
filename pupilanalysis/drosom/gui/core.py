
import os
import subprocess
import sys
import platform

from pupilanalysis.drosom.analysing import MAnalyser
from pupilanalysis.directories import ANALYSES_SAVEDIR

class Core:
    '''
    Tkinter independent functions, reusable for other GUI implementations.
    '''

    def __init__(self):
        pass


    def set_data_directory(self, data_directory):
        '''
        Update Core's knowledge about the currently selected data_directory.
        '''
        self.data_directory = data_directory


    def set_current_specimen(self, specimen_name):
        '''
        Update Core's knowledge about the currently selected specimen.
        '''
        self.current_specimen = specimen_name


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
        specimens = [fn for fn in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, fn))]
        
        if with_rois is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).are_rois_selected() == with_rois]
        
        if with_movements is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).is_measured() == with_movements]
        
        if with_correction is not None:
            specimens = [specimen for specimen in specimens if self.get_manalyser(specimen, no_data_load=True).get_antenna_level_correction() is not False]
        

        hidden = self.get_hidden()
        
        specimens = [specimen for specimen in specimens if not specimen in hidden.split(',')]

        return sorted(specimens)


    def get_manalyser(self, specimen_name, **kwargs):
        '''
        Gets manalyser for the specimen specified by the given name.
        '''
        analyser = MAnalyser(self.data_directory, specimen_name, **kwargs)
        return analyser


    def adm_subprocess(self, specimens, terminal_args, open_terminal=False):
        '''
        Starts a analyse drosom (adm) subprocess / drosom/terminal.py
        
        specimens           'current' for current selection or list of specimen names (strings)
        terminal_args        Agruments passed to the plotter
        open_terminal       If true open in a cmd window (on Windows) or lxterm (on Linux)
        '''
        
        # 1) Find python executable and wrap the filename by quation marks if spaces present
        python = sys.executable
        if ' ' in python:
            python = '"' + python + '"'

       
        # 2) Find the full path to the adm Python file in the pupil root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        pyfile = os.path.join(root, 'drosom/terminal.py')
        
        # Check for spaces in the filename. If there are spaces in the filename,
        # we have to encapsulate the filename by quation marks
        if ' ' in pyfile:
            pyfile = '"' + pyfile + '"'

        # 3) Get the specimen directoires (full paths separated by space)
        if specimens == 'current':
            manalysers = [self.get_manalyser(self.current_specimen)]
        else:
            manalysers = [self.get_manalyser(name) for name in specimens]
        specimen_directories = ' '.join(['"'+manalyser.get_specimen_directory()+'"' for manalyser in manalysers])
        

        arguments = '{} {}'.format(specimen_directories, terminal_args)
        
        command = '{} {} {} &'.format(python, pyfile, arguments)
        
        if open_terminal:
            if platform.system() == 'Linux':
                command = 'lxterm -e ' + command
            elif platform.system() == 'Windows':
                command = 'start /wait ' + command
            else:
                raise OSError('Operating system not supported by pupil?')

        print(command)
        
        subprocess.run(command, shell=True)

    def set_hidden(self, hidestring):
        '''
        hidesting       "specimen1,specimen2,..."
        '''
        
        hidelist = os.path.join(ANALYSES_SAVEDIR, 'hidelist.txt')
        with open(hidelist, 'w') as fp:
            fp.write(hidestring)


    def get_hidden(self):
        '''
        Returns hidestring, see set_hidden for more.
        '''
        hidelist = os.path.join(ANALYSES_SAVEDIR, 'hidelist.txt')
        try:
            with open(hidelist, 'r') as fp:
                line = fp.read()
        except FileNotFoundError:
            line = ''
        return line
