'''
General methods common for both DrosoM and DrosoX.
'''

import os
import json
from os import listdir
from os.path import isdir, join


from gonioanalysis.directories import GONIODIR


def simple_select(list_of_strings):
    '''
    Simple command line user interface for selecting a string
    from a list of many strings.
    
    Returns the string selected.
    '''

    for i_option, option in enumerate(list_of_strings):
        print('{}) {}'.format(i_option+1, option))

    while True:
        sel = input('Type in selection by number: ')

        try:
            sel = int(sel)
        except TypeError:
            print('Please input a number')
            continue

        if 1 <= sel <= len(list_of_strings):
            return list_of_strings[sel-1]




class SpecimenGroups:
    '''
    Managing specimens into groups. 
    '''

    def __init__(self):
        self.groups = {}
        
        self.load_groups()

    def load_groups(self):
        try:
            with open(os.path.join(GONIODIR, 'specimen_groups.txt'), 'r') as fp:
                self.groups = json.load(fp)
        except FileNotFoundError:
            print('No specimen groups')

    def save_groups(self):
        '''
        Save groups but only if groups has some members (does not save an empty dict)
        '''
        if len(self.groups.keys()) >= 0:
            with open(os.path.join(GONIODIR, 'specimen_groups.txt'), 'w') as fp:
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
    

    def parse_specimens(self, user_input):
        '''
        Parse user input to get manalyser names.

        Arguments
        ---------
        *user_input : string
            Comma separated list of specimen names or indices of self.folders,
            or a group name.
        
        Raises ValueError if unable to parse.
        '''
        
        # 1) If user supplies a specimen group name
        if user_input in self.groups.get_groups().keys():
            user_input = ','.join(self.groups.get_specimens(user_input))

            sel_keys = [os.path.basename(x) for x in user_input.split(',')]
            selections = [folder for folder in self.folders if os.path.basename(folder) in sel_keys]
            
            if len(selections) == len(sel_keys):
                print('Selecting by group.')
            else:
                print('Group is invalid; Specified specimens do not exist')
         
        else:
            # 2) Otherwise first try if indices to self.folders
            try:
                sel_indices = [int(i) for i in user_input.split(',')]
                selections = [self.filt_folders[i] for i in sel_indices]
            except IndexError:
                print('One of the given numbers goes over limits, try again.')
            
            # 3) Next try if specifying by specimen names
            except ValueError:
                print('Not number values given, trying with base names')
                
                sel_keys = [os.path.basename(x) for x in user_input.split(',')]
                selections = [folder for folder in self.folders if os.path.basename(folder) in sel_keys]
                
                if len(selections) == len(sel_keys):
                    print('Worked.')
                else:
                    print('Did not work, try again.')
        
        try:
            selections
        except:
            ValueError("parse_specimens unable to process {}".format(user_input))

        return selections
        

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
        self.filt_folders = [f for f in self.folders if
                os.path.split(f)[1].startswith(startswith) and os.path.split(f)[1].endswith(endswith) and contains in os.path.split(f)[1]]
        
        folders = self.filt_folders

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
            else:
                try:
                    selections = self.parse_specimens(user_input)
                    break
                except ValueError:
                    pass

        print('\nSelected {}\n'.format(selections))
        
        self.groups.save_groups()
    
        return selections


