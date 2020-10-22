'''
General methods common for both DrosoM and DrosoX.
'''

import os
import json
from os import listdir
from os.path import isdir, join


from pupilanalysis.directories import CODE_ROOTDIR
from pupilanalysis.cli import simple_select

class SpecimenGroups:
    '''
    Managing specimens into groups. 
    '''

    def __init__(self):
        self.groups = {}
        
        self.load_groups()

    def load_groups(self):
        try:
            with open(os.path.join(CODE_ROOTDIR, 'specimen_groups.txt'), 'r') as fp:
                self.groups = json.load(fp)
        except FileNotFoundError:
            print('No specimen groups')

    def save_groups(self):
        with open(os.path.join(CODE_ROOTDIR, 'specimen_groups.txt'), 'w') as fp:
            json.dump(self.groups, fp)

      
    def new_group(self, group_name, *specimens):
        self.groups[group_name] = [*specimens]
    

    def get_groups(self):
        '''
        Returns the groups dictionary
        '''
        return self.groups

    def get_specimens(self, group_name):
        return self.groups[group_name]


class DrosoSelect:
    '''
    Selecting a Droso folder based on user input or programmatically.

    Folder has to start with "Droso".

    TODO:
    - add programmatic selection methods of folders
    '''

    def __init__(self, datadir=None):
        '''
        datadir     Where the different droso folder are in
        '''
        if datadir is None:
            self.path = input("Input data directory >> ")
        else:
            self.path = datadir

        folders = [fn for fn in os.listdir(self.path) if isdir(join(self.path, fn))]
        self.folders = [os.path.join(self.path, fn) for fn in folders]
 
        self.groups = SpecimenGroups()
        

    def ask_user(self, startswith='', endswith='', contains=''):
        '''
        In terminal, ask user to select a Droso folder and can perform simple
        filtering of folders based on folder name.

        INPUT ARGUMENTS         DESCRIPTION
        startswith              Folder's name has to start with this string
        endswith                Folder's name has to have this string in the end
        contains                Folder's name has to have this string somewhere

        RETURNS                 A list of directories (strings) that the user selected?
        '''
        available_commands = ['new_group', 'list_groups', ]

        # Filtering of folders based on their name: startswith, endswith, and contains
        folders = [f for f in self.folders if
                os.path.split(f)[1].startswith(startswith) and os.path.split(f)[1].endswith(endswith) and contains in os.path.split(f)[1]]

        print('\nSelect a Droso folder (give either number or drosoname, to select many comma split)')
        for i, folder in enumerate(folders):
            print("  {}) {}".format(i, folder))
        
        print('Type help for additional commands')

        while True:
            user_input = input('>> ')

            # 
            splitted = user_input.split(' ')
            if splitted[0] == 'help':
                print('Following commands are avaible')
                for cmd in available_commands:
                    print('  '+cmd)
            elif splitted[0] == 'new_group':
                self.groups.new_group(*splitted[1:])
            elif splitted[0] == 'list_groups':
                print(self.groups.get_groups())
            
            elif user_input in self.groups.get_groups().keys():
                user_input = ','.join(self.groups.get_specimens(user_input))

                sel_keys = [os.path.basename(x) for x in user_input.split(',')]
                selections = [folder for folder in self.folders if os.path.basename(folder) in sel_keys]
                
                if len(selections) == len(sel_keys):
                    print('Selecting by group.')
                    break
                else:
                    print('Group is invalid; Specified specimens do not exist')
             

            else:

                try:
                    sel_indices = [int(i) for i in user_input.split(',')]
                    selections = [folders[i] for i in sel_indices]
                    break
                except IndexError:
                    print('One of the given numbers goes over limits, try again.')
                except ValueError:
                    
                    print('Not number values given, trying with base names')
                    
                    sel_keys = [os.path.basename(x) for x in user_input.split(',')]
                    selections = [folder for folder in self.folders if os.path.basename(folder) in sel_keys]
                    
                    if len(selections) == len(sel_keys):
                        print('Worked.')
                        break
                    else:
                        print('Did not work, try again.')
                 
        print('\nSelected {}\n'.format(selections))
        
        self.groups.save_groups()
    
        return selections


