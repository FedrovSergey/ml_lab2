import tkinter as tk
from tkinter import Toplevel

from Forms import DrawingApp

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
