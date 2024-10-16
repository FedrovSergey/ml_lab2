import tkinter as tk
from .NeuroNet import NeuralNetworkWindow
import os
from dotenv import load_dotenv


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Окно распознания рисунков")
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
        self.recognition_button = tk.Button(frame, text="Распознать", command=self.picture_recognition)
        self.recognition_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.neural_network_button = tk.Button(frame, text="Окно нейросети", command=self.open_neural_network_window)
        self.neural_network_button.pack(side=tk.LEFT, padx=5)
        # Text box для вывода
        self.output_text = tk.StringVar()
        self.output_text.set("")  # Initial empty string
        font = ('Helvetica', 30)  # Пример размера шрифта
        self.output_field = tk.Entry(frame, textvariable=self.output_text, state='disabled', width=40, font = font)
        self.output_field.pack(side=tk.BOTTOM, padx=5)

    def paint(self, event):
        # Задание размера карандаша
        pen_size = 5  # Пример размера карандаша в пикселях
        x1, y1 = (event.x - pen_size), (event.y - pen_size)
        x2, y2 = (event.x + pen_size), (event.y + pen_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def picture_recognition(self):
        # Stub function for picture recognition
        #print("Picture recognition function called.")
        self.output_text.set("Recognition result goes here!")  # Example text for testing

    def clear_canvas(self):
        self.canvas.delete("all")

    def open_neural_network_window(self):
        NeuralNetworkWindow(self.master)
