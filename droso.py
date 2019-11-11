'''
General methods common for both DrosoM and DrosoX.
'''

import os
from os import listdir
from os.path import isdir, join

from pupil.directories import DROSO_DATADIR

class DrosoSelect:
    '''
    Selecting a Droso folder based on user input or programmatically.

    Folder has to start with "Droso".

    TODO:
    - add programmatic selection methods of folders
    '''

    def __init__(self):
        self.path = DROSO_DATADIR
        
        folders = [fn for fn in os.listdir(self.path) if isdir(join(self.path, fn))]
        self.folders = [os.path.join(self.path, fn) for fn in folders if fn.startswith('Droso')]


    def askUser(self, startswith='', endswith='', contains=''):
        '''
        In terminal, ask user to select a Droso folder and can perform simple
        filtering of folders based on folder name.

        INPUT ARGUMENTS         DESCRIPTION
        startswith              Folder's name has to start with this string
        endswith                Folder's name has to have this string in the end
        contains                Folder's name has to have this string somewhere

        RETURNS                 A list of directories (strings) that the user selected?
        '''

        # Filtering of folders based on their name: startswith, endswith, and contains
        folders = [f for f in self.folders if
                os.path.split(f)[1].startswith(startswith) and os.path.split(f)[1].endswith(endswith) and contains in os.path.split(f)[1]]

        print('\nSelect a Droso folder (give either number or drosoname, to select many comma split)')
        for i, folder in enumerate(folders):
            print("  {}) {}".format(i, folder))

        while True:
            user_input = input('>> ')
            try:
                sel_indices = [int(i) for i in user_input.split(',')]
                selections = [folders[i] for i in sel_indices]
                break
            except IndexError:
                print('One of the given numbers goes over limits, try again.')
            except ValueError:
                
                if user_input == 'best':
                    user_input = 'DrosoM14,DrosoM16,DrosoM17,DrosoM18,DrosoM19,DrosoM20,DrosoM22,DrosoM23'

                print('Not number values given, trying with base names')
                
                sel_keys = [os.path.basename(x) for x in user_input.split(',')]
                selections = [folder for folder in self.folders if os.path.basename(folder) in sel_keys]
                
                if len(selections) == len(sel_keys):
                    print('Worked.')
                    break
                else:
                    print('Did not work, try again.')
   
        print('\nSelected {}\n'.format(selections))
        
        return selections
