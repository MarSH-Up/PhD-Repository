import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from BilinearModel_fNIRS import fNIRS_Process

# Define paths to easily import custom modules.
# Assuming the script is located two directories deep from the root of the project.
current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))

# Adding the root directory to the system path
sys.path.append(root_directory)

import numpy as np
from BilinearModel_Plots import *
from BilinerModel_Noises import awgn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Parameters.Parameters import Parameters


class ParameterInputApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("fNIRS Parameter Input")

        # Button to load parameters from a file
        self.load_button = ttk.Button(
            self, text="Load Parameters from File", command=self.load_parameters
        )
        self.load_button.pack(pady=20)

        # Ask for number of regions
        self.region_label = ttk.Label(self, text="Number of Regions:")
        self.region_label.pack(pady=10)

        self.region_entry = ttk.Entry(self)
        self.region_entry.pack(pady=10)

        self.submit_button = ttk.Button(
            self, text="Submit", command=self.generate_input_fields
        )
        self.submit_button.pack(pady=20)

    def load_parameters(self):
        filepath = filedialog.askopenfilename(
            title="Select a Parameter File",
            filetypes=(("Python files", "*.py"), ("All files", "*.*")),
        )
        if not filepath:
            return

        # Extract the Parameters dictionary from the file
        with open(filepath, "r") as file:
            exec(file.read())

        # Check if Parameters is defined in the file
        if "Parameters" not in locals():
            messagebox.showerror(
                "Error",
                "The selected file does not define a valid Parameters dictionary.",
            )
            return

        # Use the extracted Parameters to process the data and plot the results
        self.process_data_with_parameters(locals()["Parameters"])

    def generate_input_fields(self):
        try:
            self.num_regions = int(self.region_entry.get())
        except ValueError:
            messagebox.showerror(
                "Error", "Please enter a valid integer for number of regions."
            )
            return

        # Destroy previous widgets
        self.region_label.destroy()
        self.region_entry.destroy()
        self.submit_button.destroy()

        self.entries = {}

        # Generate matrix input fields based on number of regions
        for matrix_name in ["A", "B1", "B2", "C"]:
            frame = ttk.LabelFrame(self, text=matrix_name)
            frame.pack(pady=10, padx=10, fill=tk.X)

            matrix_entries = []
            for i in range(self.num_regions):
                row_entries = []
                for j in range(self.num_regions):
                    entry = ttk.Entry(frame, width=5)
                    entry.grid(row=i, column=j, padx=5, pady=5)
                    row_entries.append(entry)
                matrix_entries.append(row_entries)

            self.entries[matrix_name] = matrix_entries

        # Other parameters
        for param in ["freq", "step", "actionTime", "restTime", "cycles"]:
            frame = ttk.Frame(self)
            frame.pack(pady=10, padx=10, fill=tk.X)

            label = ttk.Label(frame, text=param)
            label.pack(side=tk.LEFT)

            entry = ttk.Entry(frame)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

            self.entries[param] = entry

        # Process button
        self.process_button = ttk.Button(
            self, text="Process", command=self.process_data
        )
        self.process_button.pack(pady=20)

    def process_data_with_parameters(self, Parameters):
        # Call your fNIRS_Process function with the Parameters
        U_stimulus, timestamps, Z, dq, dh, Y = fNIRS_Process(Parameters)

        # Initialize the plotting layout
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))

        # Plotting functions from BilinearModel_Plots
        plot_Stimulus(U_stimulus, timestamps, fig, ax1)
        plot_neurodynamics(Z, timestamps, fig, ax2)
        plot_DHDQ(dq, dh, timestamps, fig, ax3)
        plot_Y(Y, timestamps, fig, ax4)

        # Adding noise to the signal and plotting it
        noisy_signal = awgn(Y, 5, "measured")
        plot_Y(noisy_signal, timestamps, fig, ax5)

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(pady=20)
        canvas.draw()

    def process_data(self):
        Parameters = {}

        # Fetch matrix values
        for matrix_name in ["A", "B1", "B2", "C"]:
            matrix_entries = self.entries[matrix_name]
            matrix_values = []
            for row_entries in matrix_entries:
                row_values = [self.safe_eval(entry.get()) for entry in row_entries]
                matrix_values.append(row_values)
            Parameters[matrix_name] = np.array(matrix_values)

        # Construct B matrix from B1 and B2
        B = np.zeros((Parameters["B1"].shape[0], Parameters["B1"].shape[1], 2))
        B[:, :, 0] = Parameters["B1"]
        B[:, :, 1] = Parameters["B2"]
        Parameters["B"] = B

        # Fetch other parameters
        for param in ["freq", "step", "actionTime", "restTime", "cycles"]:
            Parameters[param] = self.safe_eval(self.entries[param].get())

        # Set P_SD as constant
        Parameters["P_SD"] = np.array(
            [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
        )

        # Call your fNIRS_Process function with the Parameters
        U_stimulus, timestamps, Z, dq, dh, Y = fNIRS_Process(Parameters)

        # Initialize the plotting layout
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 18))

        # Plotting functions from BilinearModel_Plots
        plot_Stimulus(U_stimulus, timestamps, fig, ax1)
        plot_neurodynamics(Z, timestamps, fig, ax2)
        plot_DHDQ(dq, dh, timestamps, fig, ax3)
        plot_Y(Y, timestamps, fig, ax4)

        # Adding noise to the signal and plotting it
        noisy_signal = awgn(Y, 5, "measured")
        plot_Y(noisy_signal, timestamps, fig, ax5)

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(pady=20)
        canvas.draw()

    def on_closing(self):
        # Close all plots
        plt.close("all")
        # Destroy the main application window
        self.destroy()

    def safe_eval(self, s):
        # Check if the string contains only numbers and basic arithmetic operations
        allowed_chars = set("0123456789.+-*/()")
        if set(s) <= allowed_chars:
            return eval(s)
        else:
            raise ValueError(f"Invalid input: {s}")


if __name__ == "__main__":
    app = ParameterInputApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
