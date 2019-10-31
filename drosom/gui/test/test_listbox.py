
import tkinter as tk

from pupil.drosom.gui.elements import Listbox
       

def main():
    root = tk.Tk()
    frame = Listbox(root, ['test1', 'test2', 'test3'], print)
    frame.grid()
    root.mainloop()

if __name__ == "__main__":
    main()
