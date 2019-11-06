
import os
import subprocess
import sys

from pupil.drosom.analysing import MAnalyser


class Core:
    '''
    Tkinter independent functions, reusable for other GUI implementations.
    '''

    def __init__(self):
        pass

    def set_data_directory(self, data_directory):
        self.data_directory = data_directory
    
    def set_current_specimen(self, specimen_name):
        self.current_specimen = specimen_name
    
    def list_specimens(self):
        '''
        List specimens in the data directory. May contain bad folders also (no check for contents)
        '''
        specimens = [fn for fn in os.listdir(self.data_directory) if os.path.isdir(os.path.join(self.data_directory, fn))]
        return sorted(specimens)


    def get_manalyser(self, specimen_name):
        '''
        Gets manalyser for the specimen specified by the given name.
        '''
        analyser = MAnalyser(self.data_directory, specimen_name)
        return analyser


    def adm_subprocess(self, specimens, terminal_args):
        '''
        Starts a analyse drosom (adm) subprocess / drosom/terminal.py
        
        specimens           'current' for current selection or list of specimen names (strings)
        terminal_args        Agruments passed to the plotter
        '''
        
        # Find the full path to the adm Python file in the pupil root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        pyfile = os.path.join(root, 'adm')
        

        # Get the specimen directoires (full paths separated by space)
        if specimens == 'current':
            manalysers = [self.get_manalyser(self.current_specimen)]
        else:
            manalysers = [self.get_manalyser(name) for name in specimens]
        specimen_directories = ' '.join([manalyser.get_specimen_directory() for manalyser in manalysers])
        
        arguments = '{} {}'.format(specimen_directories, terminal_args)
        
        python = sys.executable

        command = '{} {} {} &'.format(python, pyfile, arguments)
        print(command)
        
        subprocess.run(command, shell=True)

