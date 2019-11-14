
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

    def __init__(self, tk_root, manalysers):
        '''
        tk_root         Tkinter root object, needed for scheduling events with after-method
        manalysers      List of manalysers
        '''
        self.root = tk_root

        self.i_manalyser = -1
        self.manalysers = manalysers
        self.manalyser = manalysers[0]
        
        p = threading.Thread(target=self.run)
        p.start()
        
        self.processes = []


    def run(self):
        self.top = tk.Toplevel()
        self.top.title('Measuring movement...')

        self.oldout = sys.stdout
        sys.stdout = io.StringIO()

        BufferShower(self.top, sys.stdout).grid()
        self.cancel_button = tk.Button(self.top, text='Cancel', command=self.on_cancel)
        self.cancel_button.grid()
        
        self.check_finished()

    def _run_next_manalyser(self):
        '''
        Set next manalyser to work or return False if none left.
        '''
        self.i_manalyser += 1
        
        if self.i_manalyser == len(self.manalysers):
            return False

        self.manalyser = self.manalysers[self.i_manalyser]
        

        p = threading.Thread(target=self.manalyser.measure_both_eyes) 
        p.start()
        self.processes.append(p)
    
        return True


    def is_finished(self):
        '''
        Returns true if the spawned threads have finished
        '''
        if not all([stopped.is_alive() for stopped in self.processes]):
            return True
        elif self.processes == []:
            return True
        return False
    

    def check_finished(self):
        '''

        '''
        if self.is_finished():
            manalysers_left = self._run_next_manalyser()
        else:
            manalysers_left = True

        if manalysers_left:
            self.root.after(1000, self.check_finished)
        else:
            self.cancel_button.config(text='Ok')


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
