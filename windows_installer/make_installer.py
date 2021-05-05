'''
A script to make an all-in-one Windows installer for gonio-analyis using
pip, pynsist and NSIS.

You have to
    - be on the Windows platfrom (64-bit)
    - have pynsist and NSIS installed
    - have the same major.minor version of Python as PYTHONVERSION in here

Because the wheels embedded in the installer are fetched from PyPi, this script
can make installers only for PyPi released versions of gonio-analysis.

Attributes
----------
GONIOVERSION : string
    Version of the Gonio Analysis to use.
    By default, it is assumed that we are in a git work copy.
PYTHONVERSION : string
    Version of the Python interpreter to use
'''

import os
import sys
import shutil

try:
    # Version number to __version__ variable
    exec(open("../gonioanalysis/version.py").read())
except:
    __version__ == input('gonioanalysis version use (exmpl. 0.1.2) >>')

GONIOVERSION = __version__
PYTHONVERSION = '{}.{}.{}'.format(*sys.version_info[0:3])

def fetch_wheels():
    
    if os.path.isdir('wheels'):
        shutil.rmtree('wheels')
    os.makedirs('wheels')
    
    os.chdir('wheels')
    os.system('pip download gonio-analysis=='+GONIOVERSION)
    os.chdir('..')
    


def build(gonioversion, pythonversion):
    
    os.system('get_tkdeps.py')
    
    fetch_wheels()
    wheels = [os.path.join('wheels', fn) for fn in os.listdir('wheels') if fn.endswith('.whl')]
    
    str_wheels = '\n '.join(wheels)

    moveversion = [fn for fn in wheels if 'movemeter-' in fn][0]
    moveversion = moveversion.split('-')[1]
    
    cfg_file = []
    with open('pynsist_template.cfg', 'r') as fp:
        for line in fp:
            edited = line.replace('GONIOVERSION', gonioversion)
            edited = edited.replace('PYTHONVERSION', pythonversion)
            edited = edited.replace('LOCAL_WHEELS', str_wheels)
            edited = edited.replace('MOVEVERSION', moveversion)
            cfg_file.append(edited)

    with open('pynsist_temp.cfg', 'w') as fp:
        for line in cfg_file:
            fp.write(line)

    print(cfg_file)

    os.system('pynsist pynsist_temp.cfg')
    #shutil.rmtree('wheels')

def main():
    build(GONIOVERSION, PYTHONVERSION)

if __name__ == "__main__":
    main()

