'''
Central settings for save/load directories.
'''

import os
import platform

CODE_ROOTDIR = os.path.dirname(os.path.realpath(__file__))
USER_HOMEDIR = os.path.expanduser('~')

if platform.system() == "Windows":
    PUPILDIR = os.path.join(USER_HOMEDIR, 'PupilAnalysis')
else:
    PUPILDIR = os.path.join(USER_HOMEDIR, '.pupilanalysis')


ANALYSES_SAVEDIR = os.path.join(PUPILDIR, 'final_results')
PROCESSING_TEMPDIR = os.path.join(PUPILDIR, 'intermediate_data')
PROCESSING_TEMPDIR_BIGFILES = os.path.join(PUPILDIR, 'intermediate_bigfiles')


# DIRECTORIES THAT HAVE TO BE CREATED
ALLDIRS= {'ANALYSES_SAVEDIR': ANALYSES_SAVEDIR,
        'PROCESSING_TEMPDIR': PROCESSING_TEMPDIR,
        'PROCESSING_TEMPDIR_BIGFILES': PROCESSING_TEMPDIR_BIGFILES}


def print_directories():

    print('These are the directories')
    
    for key, item in ALLDIRS.items():
        print('{} {}'.format(key, item))



def cli_ask_creation(needed_directories):
    '''
    Short command line yes/no.
    '''
    print("The following directories have to be created")

    for directory in needed_directories:
        print("  {}".format(directory))

    print("\nIs this okay? (yes/no)")

    while True:
        selection = input(">> ").lower()

        if selection == "yes":
            return True
        elif selection == "no":
            return False
        else:
            print("Choice not understood, please type yes or no")



def directories_check(ui=cli_ask_creation):
    '''
    Perform a check that the saving directories exist.
    
    ui      Callable that returns True if user wants to create
                the directories or False if not.
                As the first argument it gets the list of to be created directories

                If ui is not callable, raise an error.
    '''
    non_existant = []
    
    for key, item in ALLDIRS.items():

        if not os.path.exists(item):
            non_existant.append(item)

    # If some directories are not created, launch an input requiring
    # user interaction.
    if non_existant:
        if callable(ui):
            if ui(non_existant) == True:
                for directory in non_existant:
                    os.makedirs(directory, exist_ok=True)
            else:
                raise NotImplementedError("Reselecting directories in UI not yet implemented")
        else:
            raise OSError("All directories not created and ui is not callable")



if __name__ == "__main__":
    print_directories()
else:
    directories_check()



