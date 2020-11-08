
import tkinter as tk

class RepetitionSelector(tk.Frame):

    def __init__(self, tk_master, RecordingPlotter, update_command):
        '''
        
        update_command      Callable that updates the plots, no input arguments
        '''

        self.update_command = update_command

        tk.Frame.__init__(self, tk_master)


        self.plotter = RecordingPlotter

        self.all = tk.Button(self, text='Show all', command=lambda: self.move_selection(None))
        self.all.grid(row=0, column=0)

        self.previous = tk.Button(self, text='Previous', command=lambda: self.move_selection(-1))
        self.previous.grid(row=0, column=1)

        self.next = tk.Button(self, text='Next', command=lambda: self.move_selection(1))
        self.next.grid(row=0, column=2)

    def move_selection(self, direction):
        '''
        None        sets plotter to show all repeats
        1 or -1     Move to next/previous repetition
        '''

        if direction == None:
            self.plotter.i_repeat = None
        else:
            if self.plotter.i_repeat == None:
                self.plotter.i_repeat = 0
            else:
                self.plotter.i_repeat += direction

            if self.plotter.i_repeat < 0:
                self.plotter.i_repeat = 0
            elif self.plotter.i_repeat >= self.plotter.N_repeats:
                self.plotter.i_repeat = self.plotter.N_repeats -1

        print(self.plotter.i_repeat)

        if self.update_command:
            self.update_command()
