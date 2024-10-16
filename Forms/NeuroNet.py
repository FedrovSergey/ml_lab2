import tkinter as tk
from tkinter import messagebox, Toplevel

from Forms.Metric import graths
from Service import train, validation
from Service.Train import train_validate_for_metrics


class NeuralNetworkWindow:
    def __init__(self, master):
        self.master = Toplevel(master)
        self.master.title("Нейросеть")
        self.master.geometry("400x256")
        self.master.grab_set()

        # Create frames for different areas with borders
        self.epochs_frame = tk.LabelFrame(self.master, text="Ввод данных", padx=10, pady=10)
        self.epochs_frame.place(relx=0.5, rely=0.1, anchor='n')

        self.learning_frame = tk.LabelFrame(self.master, text="Обучение", padx=10, pady=10)
        self.learning_frame.place(relx=0.5, rely=0.4, anchor='n')

        self.validation_frame = tk.LabelFrame(self.master, text="Валидация", padx=10, pady=10)
        self.validation_frame.place(relx=0.3, rely=0.7, anchor='n')

        # Metrics frame
        self.metrics_frame = tk.LabelFrame(self.master, text="Метрики", padx=10, pady=10)
        self.metrics_frame.place(relx=0.7, rely=0.7, anchor='n')

        # Epochs input
        tk.Label(self.epochs_frame, text="количество эпох:").pack(side='left')
        self.epochs_entry = tk.Entry(self.epochs_frame)
        self.epochs_entry.pack(side='left')

        # Speed learning input
        tk.Label(self.epochs_frame, text="Скорость обучения:").pack(side='left')
        self.speed_entry = tk.Entry(self.epochs_frame)
        self.speed_entry.pack(side='bottom')

        # Buttons for learning processes
        learn_new_button = tk.Button(self.learning_frame, text="Обучить с нуля", command=self.train_from_scratch)
        learn_new_button.pack(side='left', padx=5)

        learn_button = tk.Button(self.learning_frame, text="Дообучить", command=self.retrain)
        learn_button.pack(side='left', padx=5)

        # Validation button
        validate_button = tk.Button(self.validation_frame, text="Протестировать", command=self.validate_model)
        validate_button.pack(padx=5, pady=5)  # Center the button within the frame

        # Graphs button in metrics frame
        graphs_button = tk.Button(self.metrics_frame, text="Построить графики", command=self.show_graphs)
        graphs_button.pack(padx=5, pady=5)  # Center the button within the frame

    def show_graphs(self):
        graths()
    def train_from_scratch(self):
        if self.confirm_action("Are you sure you want to start learning from scratch?"):
            epochs = self.get_epochs()
            if epochs is not None:
                print(f"Train from scratch for {epochs} epochs.")
                # Add your training logic here
                #train(epochs, True)
                #для вывода данных нужных для графиков в нужной последовательности
                train_validate_for_metrics(epochs, True)

    def retrain(self):
        if self.confirm_action("Are you sure you want to start learning without zeroing weights?"):
            epochs = self.get_epochs()
            if epochs is not None:
                print(f"Learning for {epochs} epochs without zeroing weights.")
                # Add your retraining logic here
                train(epochs, False)

    def validate_model(self):
        # Validation logic can go here
        print("Validation process started...")
        epochs = self.get_epochs()
        validation(epochs)
    def get_epochs(self):
        try:
            epochs = int(self.epochs_entry.get())
            return epochs
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of epochs.")
            return None

    def confirm_action(self, message):
        return messagebox.askyesno("Confirm Action", message)

