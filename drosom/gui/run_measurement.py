
import io
import sys
import threading

import tkinter as tk
from tk_steroids.elements import BufferShower

class MeasurementWindow:
    '''
    Sets the manalyser to do the movement measurements and
    opens a window showing the progress.

    Because this sets stdout to a StringIO temporalily, it's best
    to run this in a subprocess.
    '''

    def __init__(self, manalyser):
        self.manalyser = manalyser
        
        p = threading.Thread(target=self.run)
        p.start()
        
        self.processes = []


    def run(self):
        self.top = tk.Toplevel()
        self.top.title('Measuring movement...')

        self.oldout = sys.stdout
        sys.stdout = io.StringIO()

        BufferShower(self.top, sys.stdout).grid()
        tk.Button(self.top, text='Cancel', command=self.on_cancel).grid()


        for eye in ['left', 'right']:
            p = threading.Thread(target=self.manalyser.measure_movement, args=(eye,)) 
            p.start()
            self.processes.append(p)


    def on_cancel(self):
        self.manalyser.stop()

        # Destroy the window if everything analysed
        if not all([stopped.is_alive() for stopped in self.processes]):
            self.top.destroy()
            sys.stdout = self.oldout



def main():
    '''
    Read data directory and specimen name from sys.argv
    and run movement measurement for that manalyser.
    '''
    pass

if __name__ == main():
    main()
