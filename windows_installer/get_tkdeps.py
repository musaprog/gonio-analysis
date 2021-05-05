
import os
import sys
import shutil

def main():
    workdir = os.getcwd()
    pdir = os.path.dirname(sys.executable)

    shutil.copytree(os.path.join(pdir, 'tcl'), 'lib')
    os.makedirs('pynsist_pkgs', exist_ok=True)

    copyfiles = [os.path.join(pdir, 'DLLs', fn) for fn in ['_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll']]
    copyfiles.append(os.path.join(pdir, 'libs', '_tkinter.lib'))

    for origfile in copyfiles:
        newfile = os.path.join('pynsist_pkgs', os.path.basename(origfile))
        shutil.copy(origfile, newfile)

if __name__ == "__main__":
    main()
