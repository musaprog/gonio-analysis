'''
A script to make an all-in-one Windows installer for pupil-analyis using
pip, pynsist and NSIS.

You have to
    - be on the Windows platfrom (64-bit)
    - have pynsist and NSIS installed
    - have the same major.minor version of Python as PYTHONVERSION in here

Because the wheels embedded in the installer are fetched from PyPi, this script
can make installers only for PyPi released versions of pupil-analysis.

Attributes
----------
PUPILVERSION : string
    Version of the Pupil Analysis to use.
    By default, it is assumed that we are in a git work copy.
PYTHONVERSION : string
    Version of the Python interpreter to use
'''

import os
import shutil

try:
    # Version number to __version__ variable
    exec(open("../pupilanalysis/version.py").read())
except:
    __version__ == input('pupilanalysis version use (exmpl. 0.1.2) >>')

PUPILVERSION = __version__
PYTHONVERSION = '3.8.6'

def fetch_wheels():
    
    if os.path.isdir('wheels'):
        shutil.rmtree('wheels')
    os.makedirs('wheels')
    
    os.chdir('wheels')
    os.system('pip download pupil-analysis=='+PUPILVERSION)
    os.chdir('..')
    


def build(pupilversion, pythonversion):
    
    os.system('get_tkdeps.py')
    
    fetch_wheels()
    wheels = [os.path.join('wheels', fn) for fn in os.listdir('wheels') if fn.endswith('.whl')]
    
    str_wheels = '\n '.join(wheels)
    
    cfg_file = []
    with open('pynsist_template.cfg', 'r') as fp:
        for line in fp:
            edited = line.replace('PUPILVERSION', pupilversion)
            edited = edited.replace('PYTHONVERSION', pythonversion)
            edited = edited.replace('LOCAL_WHEELS', str_wheels)
            cfg_file.append(edited)

    with open('pynsist_temp.cfg', 'w') as fp:
        for line in cfg_file:
            fp.write(line)

    print(cfg_file)

    os.system('pynsist pynsist_temp.cfg')
    #shutil.rmtree('wheels')

def main():
    build(PUPILVERSION, PYTHONVERSION)

if __name__ == "__main__":
    main()

