
import sys, os

import tkinter as tk
from tkinter import filedialog, simpledialog

# Add the directory above the pupil folder to path because
# other needed packages may lay there
sys.path.append(os.path.dirname(os.getcwd()))

from imalyser.movement.tk_measure import runSingle

def main():
    tk_window = tk.Tk()
    tk_window.withdraw()

    specimen_folder = filedialog.askdirectory(title='Select data folder(s)', parent=tk_window)
    
    tk_window.destroy()

    if specimen_folder is None:
        return 0

    runSingle(folder=specimen_folder, chdir='../RESULTS/tmp/measure_single')

if __name__ == "__main__":
    main()
