'''
A script to autogenerate .cmd files for Windows in the bin.
'''

import os

def main():

    os.chdir('bin')
    
    pyfiles = [fn for fn in os.listdir() if fn.endswith('.py')]

    for pyfile in pyfiles:
        # Here assuming there's only one . in the filenames
        cmd_fn = pyfile.split('.')[0] + '.cmd'
        with open(cmd_fn, 'w') as fp:
            fp.write('python {}'.format(pyfile))

if __name__ == "__main__":
    main()
