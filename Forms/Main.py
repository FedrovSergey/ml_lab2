import tkinter as tk
from .NeuroNet import NeuralNetworkWindow
import os
from dotenv import load_dotenv


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing Application")
        self.master.geometry("512x512")  # Set window size to 512x512 pixels

        load_dotenv()
        const_width = os.getenv('const_width')

        # Center the window on the screen
        window_width = 512
        window_height = 512
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        x_coordinate = (screen_width // 2) - (window_width // 2)
        y_coordinate = (screen_height // 2) - (window_height // 2)

        self.master.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

        # Canvas for drawing
        frame = tk.Frame(self.master)
        frame.pack(expand=True)  # Расширяем фрейм, чтобы занять доступное пространство

        # Canvas для рисования
        self.canvas = tk.Canvas(frame, width=200, height=200, bg='white')
        self.canvas.pack(padx=16, pady=16)  # Добавляем отступы для эстетики
        self.canvas.bind("<B1-Motion>", self.paint)

        # Кнопки
        self.recognition_button = tk.Button(frame, text="Recognize", command=self.picture_recognition)
        self.recognition_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.neural_network_button = tk.Button(frame, text="Neural Network", command=self.open_neural_network_window)
        self.neural_network_button.pack(side=tk.LEFT, padx=5)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def picture_recognition(self):
        # Stub function for picture recognition
        print("Picture recognition function called.")

    def clear_canvas(self):
        self.canvas.delete("all")

    def open_neural_network_window(self):
        NeuralNetworkWindow(self.master)
