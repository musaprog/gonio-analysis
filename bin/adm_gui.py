'''
Equvalent of adm but instead of being a command line tool,
use tkinter to create simple dialog windows to use
the command line tool.
'''

import os
import sys

import tkinter as tk
from tkinter import filedialog, simpledialog

# Add the directory above the pupil folder to path because
# other needed packages may lay there
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))
 
from drosom.terminal import TerminalDrosoM


def main():
    
    readme = ''
    with open(os.path.join('..', 'drosom', 'README.txt'), 'r') as fp:
        for line in fp:
            readme += line

    tk_window = tk.Tk()
    tk_window.withdraw()
    
    args = simpledialog.askstring("Pass options", readme, parent=tk_window)
    
    if args is None:
        return 0
    
    specimen_folder = filedialog.askdirectory(title='Select data folder(s)', parent=tk_window)
    
    if specimen_folder is None:
        return 0

    tk_window.destroy()

    terminal = TerminalDrosoM( custom_args=args )
    terminal.main( data_folder = [specimen_folder] )
    


if __name__ == "__main__":
    main()
