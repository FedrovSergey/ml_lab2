# import tkinter as tk
# from tkinter import messagebox, Toplevel
#
# from Service import train
#
# class NeuralNetworkWindow:
#     def __init__(self, master):
#         self.master = Toplevel(master)
#         self.master.title("Neural Network")
#         self.master.geometry("256x128")
#         self.master.grab_set()
#
#         # Create frames for different areas
#         self.epochs_frame = tk.Frame(self.master)
#         self.epochs_frame.place(relx=0.5, rely=0.3, anchor='center')
#
#         self.learning_frame = tk.Frame(self.master)
#         self.learning_frame.place(relx=0.5, rely=0.5, anchor='center')
#
#         self.actions_frame = tk.Frame(self.master)
#         self.actions_frame.place(relx=0.5, rely=0.7, anchor='center')
#
#         # Epochs input
#         tk.Label(self.epochs_frame, text="Epochs:").pack(side='left')
#         self.epochs_entry = tk.Entry(self.epochs_frame)
#         self.epochs_entry.pack(side='left')
#
#         # Buttons for learning processes
#         learn_new_button = tk.Button(self.learning_frame, text="Train from Scratch", command=self.train_from_scratch)
#         learn_new_button.pack(side='left', padx=5)
#
#         learn_button = tk.Button(self.learning_frame, text="Retrain", command=self.retrain)
#         learn_button.pack(side='left', padx=5)
#
#     def train_from_scratch(self):
#         if self.confirm_action("Are you sure you want to start learning from scratch?"):
#             epochs = self.get_epochs()
#             if epochs is not None:
#                 print(f"Train from scratch for {epochs} epochs.")
#                 train(epochs, True)
#
#     def retrain(self):
#         if self.confirm_action("Are you sure you want to start learning without zeroing weights?"):
#             epochs = self.get_epochs()
#             if epochs is not None:
#                 print(f"Learning for {epochs} epochs without zeroing weights.")
#                 train(epochs, False)
#
#
#     def get_epochs(self):
#         try:
#             epochs = int(self.epochs_entry.get())
#             return epochs
#         except ValueError:
#             messagebox.showerror("Invalid input", "Please enter a valid number of epochs.")
#             return None
#
#
#     def confirm_action(self, message):
#         return messagebox.askyesno("Confirm Action", message)


# import tkinter as tk
# from tkinter import messagebox, Toplevel
#
# class NeuralNetworkWindow:
#     def __init__(self, master):
#         self.master = Toplevel(master)
#         self.master.title("Neural Network")
#         self.master.geometry("300x200")
#         self.master.grab_set()
#
#         # Create frames for different areas
#         self.epochs_frame = tk.Frame(self.master)
#         self.epochs_frame.pack(pady=10, fill='x')
#
#         self.learning_frame = tk.Frame(self.master)
#         self.learning_frame.pack(pady=10, fill='x')
#
#         self.validation_frame = tk.Frame(self.master)
#         self.validation_frame.pack(pady=10, fill='x')
#
#         self.actions_frame = tk.Frame(self.master)
#         self.actions_frame.pack(pady=10, fill='x')
#
#         # Epochs input
#         tk.Label(self.epochs_frame, text="Epochs:").pack(side='left', padx=5)
#         self.epochs_entry = tk.Entry(self.epochs_frame)
#         self.epochs_entry.pack(side='left', padx=5)
#
#         # Buttons for learning processes
#         learn_new_button = tk.Button(self.learning_frame, text="Train from Scratch", command=self.train_from_scratch)
#         learn_new_button.pack(side='left', padx=5)
#
#         learn_button = tk.Button(self.learning_frame, text="Retrain", command=self.retrain)
#         learn_button.pack(side='left', padx=5)
#
#         # Validation button
#         validate_button = tk.Button(self.validation_frame, text="Validate", command=self.validate_model)
#         validate_button.pack(side='left', padx=5)
#
#     def train_from_scratch(self):
#         if self.confirm_action("Are you sure you want to start learning from scratch?"):
#             epochs = self.get_epochs()
#             if epochs is not None:
#                 print(f"Train from scratch for {epochs} epochs.")
#                 train(epochs, True)
#
#     def retrain(self):
#         if self.confirm_action("Are you sure you want to start learning without zeroing weights?"):
#             epochs = self.get_epochs()
#             if epochs is not None:
#                 print(f"Learning for {epochs} epochs without zeroing weights.")
#                 train(epochs, False)
#
#     def validate_model(self):
#         print("Validation process started...")
#
#     def get_epochs(self):
#         try:
#             epochs = int(self.epochs_entry.get())
#             return epochs
#         except ValueError:
#             messagebox.showerror("Invalid input", "Please enter a valid number of epochs.")
#             return None
#
#     def confirm_action(self, message):
#         return messagebox.askyesno("Confirm Action", message)
#
# # Sample implementation of train function
# def train(epochs, from_scratch):
#     if from_scratch:
#         print(f"Training from scratch for {epochs} epochs.")
#     else:
#         print(f"Retraining for {epochs} epochs.")


import tkinter as tk
from tkinter import messagebox, Toplevel

from Service import train, validation

class NeuralNetworkWindow:
    def __init__(self, master):
        self.master = Toplevel(master)
        self.master.title("Neural Network")
        self.master.geometry("400x256")
        self.master.grab_set()

        # Create frames for different areas with borders
        self.epochs_frame = tk.LabelFrame(self.master, text="Epochs Input", padx=10, pady=10)
        self.epochs_frame.place(relx=0.5, rely=0.1, anchor='n')

        self.learning_frame = tk.LabelFrame(self.master, text="Learning Options", padx=10, pady=10)
        self.learning_frame.place(relx=0.5, rely=0.4, anchor='n')

        self.validation_frame = tk.LabelFrame(self.master, text="Validation", padx=10, pady=10)
        self.validation_frame.place(relx=0.5, rely=0.7, anchor='n')

        # Epochs input
        tk.Label(self.epochs_frame, text="Epochs:").pack(side='left')
        self.epochs_entry = tk.Entry(self.epochs_frame)
        self.epochs_entry.pack(side='left')

        # Buttons for learning processes
        learn_new_button = tk.Button(self.learning_frame, text="Train from Scratch", command=self.train_from_scratch)
        learn_new_button.pack(side='left', padx=5)

        learn_button = tk.Button(self.learning_frame, text="Retrain", command=self.retrain)
        learn_button.pack(side='left', padx=5)

        # Validation button
        validate_button = tk.Button(self.validation_frame, text="Validate", command=self.validate_model)
        validate_button.pack(padx=5, pady=5)  # Center the button within the frame

    def train_from_scratch(self):
        if self.confirm_action("Are you sure you want to start learning from scratch?"):
            epochs = self.get_epochs()
            if epochs is not None:
                print(f"Train from scratch for {epochs} epochs.")
                # Add your training logic here
                train(epochs, True)

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

