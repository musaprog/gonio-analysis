import threading

import tkinter as tk


class WaitingFrame(tk.Frame):

    def __init__(self, tk_master, text):
        tk.Frame.__init__(self, tk_master)
        tk.Label(self, text=text).grid()


class WaitingWindow():
    '''
    Spans a new tkinter root window so use this only
    if there's no tkinter root beforehad.
    '''
    def __init__(self, title, text):
        
        self.root = tk.Tk()
        self.root.title(title)
        WaitingFrame(self.root, text).grid()
        self.root.update()

    def close(self):
        self.root.destroy()
