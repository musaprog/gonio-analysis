
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

    def __init__(self, tk_root, thread_targets, title='', callback_on_exit=None):
        '''
        tk_root         Tkinter root object, needed for scheduling events with after-method
        thread_targets  Callables
        '''
        self.root = tk_root
        self.title = title
        self.callback_on_exit = callback_on_exit

        #self.i_manalyser = -1
        #self.manalysers = manalysers
        #self.manalyser = manalysers[0]
        
        self.i_target = -1
        self.thread_targets = thread_targets

        #p = threading.Thread(target=self.run)
        #p.start()
        
        self.processes = []

        self.run()

    def run(self):
        self.top = tk.Toplevel()
        self.top.title(self.title)

        self.oldout = sys.stdout
        sys.stdout = io.StringIO()

        BufferShower(self.top, sys.stdout).grid()
        self.cancel_button = tk.Button(self.top, text='Cancel', command=self.on_cancel)
        self.cancel_button.grid()
        
        self.check_finished()

    def _run_next_target(self):
        '''
        Set next manalyser to work or return False if none left.
        '''
        self.i_target += 1
        
        if self.i_target == len(self.thread_targets):
            return False


        p = threading.Thread(target=self.thread_targets[self.i_target]) 
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
        Check if all the threads are finished and if there are more targets to run.
        Reschedule every 1000 ms.
        '''
        if self.is_finished():
            targets_left = self._run_next_target()
        else:
            targets_left = True

        if targets_left:
            self.root.after(1000, self.check_finished)
        else:
            self.cancel_button.config(text='Ok')


    def on_cancel(self):
        if not self.is_finished():
            self.thread_targets[self.i_target].stop()
        else:
            self.callback_on_exit()

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
