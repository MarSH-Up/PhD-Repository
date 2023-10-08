import base64
import os

import numpy as np
import pandas as pd
import streamlit as st
from BilinearModel_CSVGenerator import write_to_csv
from BilinearModel_fNIRS import fNIRS_Process
from BilinearModel_Plots import *
from BilinerModel_Noises import awgn
from matplotlib import pyplot as plt


def main():
    st.title("Bilinear Model for fNIRS")

    # Load parameters from file
    uploaded_file = st.file_uploader("Load Parameters from File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        Parameters = parse_csv_to_parameters(df)

        if Parameters:
            num_regions = Parameters["A"].shape[0]
            process_data_with_parameters(Parameters)
        else:
            st.error("The selected file does not define a valid Parameters dictionary.")
            return
    else:
        # Ask for number of regions only if no file is uploaded}
        st.markdown("<h4>Manual Description</h4>", unsafe_allow_html=True)
        num_regions = st.number_input("Number of Regions:", min_value=1, step=1)
        if st.button("Submit"):
            generate_input_fields(num_regions)


def save_csv(Y, timestamps, filename):
    # Convert Y and timestamps to CSV string
    csv_content = (
        "Timestamps," + ",".join([f"Region_{i+1}" for i in range(Y.shape[0])]) + "\n"
    )
    for j in range(Y.shape[1]):
        row = [str(timestamps[j])] + list(map(str, Y[:, j]))
        csv_content += ",".join(row) + "\n"

    # Save the CSV string to a file
    with open(filename, "w") as f:
        f.write(csv_content)


def get_csv_link(filename):
    # Convert file to base64
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Return the file as a download link
    return f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(filename)}">Download CSV</a>'


def generate_input_fields(num_regions):
    # Generate matrix input fields based on number of regions
    matrices = {}
    for matrix_name in ["A", "B1", "B2", "C"]:
        matrix = np.zeros((num_regions, num_regions))
        for i in range(num_regions):
            for j in range(num_regions):
                matrix[i][j] = st.number_input(f"{matrix_name}[{i}][{j}]", value=0.0)
        matrices[matrix_name] = matrix

    # Other parameters
    params = {}
    for param in ["freq", "step", "actionTime", "restTime", "cycles"]:
        params[param] = st.number_input(param, value=0.0)

    if st.button("Process"):
        process_data(matrices, params)


def process_data(matrices, params):
    # Construct B matrix from B1 and B2
    B = np.zeros((matrices["A"].shape[0], matrices["A"].shape[1], 2))
    B[:, :, 0] = matrices["B1"]
    B[:, :, 1] = matrices["B2"]
    matrices["B"] = B

    # Your processing logic here
    process_data_with_parameters({**matrices, **params})


def ensure_directory_exists(directory):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_data_with_parameters(Parameters):
    # Your existing logic to process data and plot

    U_stimulus, timestamps, Z, dq, dh, Y = fNIRS_Process(Parameters)

    # Initialize the plotting layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
    fig.subplots_adjust(hspace=0.5)

    # Plotting functions from BilinearModel_Plots

    plot_Stimulus(U_stimulus, timestamps, fig, ax1)
    # plot_neurodynamics(Z, timestamps, fig, ax2)
    # plot_DHDQ(dq, dh, timestamps, fig, ax3)
    plot_Y(Y, timestamps, fig, ax2)

    # Adding noise to the signal and plotting it
    noisy_signal = awgn(Y, 5, "measured")
    plot_Y(noisy_signal, timestamps, fig, ax3)

    # Display the matrices in the desired format
    st.title("Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Matrix A:")
        st.write(Parameters["A"])
        st.write("Matrix B1:")
        st.write(Parameters["B"][:, :, 0])

    with col2:
        st.write("Matrix C:")
        st.write(Parameters["C"])
        st.write("Matrix B2:")
        st.write(Parameters["B"][:, :, 1])

    # In your Streamlit code:
    if st.button("EXPORT RESULTS TO CSV"):
        st.session_state.export_csv = True

    if "export_csv" in st.session_state and st.session_state.export_csv:
        user_filename = st.text_input(
            "Enter the filename to save the CSV:", "output.csv"
        )
        directory = "data"
        ensure_directory_exists(directory)  # Ensure the directory exists
        full_filename = os.path.join(directory, user_filename)

        if st.button("Save CSV"):
            if user_filename:
                save_csv(Y, timestamps, full_filename)

                link = get_csv_link(full_filename)
                st.markdown(link, unsafe_allow_html=True)

                st.success(f"Results exported successfully to {full_filename}!")
                st.session_state.export_csv = False  # Reset the state
            else:
                st.warning("Please provide a filename to save the CSV.")

    # Display the plot in Streamlit
    st.title("Results")
    st.pyplot(fig)


def parse_csv_to_parameters(df):
    matrices = ["A", "B1", "B2", "C", "P_SD"]
    Parameters = {}

    for matrix in matrices:
        matrix_df = df[df["Parameter"] == matrix]
        size = int(max(matrix_df["Row"]))
        mat = np.zeros((size, size))
        for _, row in matrix_df.iterrows():
            mat[int(row["Row"]) - 1][int(row["Col"]) - 1] = row["Value"]
        Parameters[matrix] = mat

    # Handle B matrix
    B = np.zeros((Parameters["A"].shape[0], Parameters["A"].shape[1], 2))
    B[:, :, 0] = Parameters["B1"]
    B[:, :, 1] = Parameters["B2"]
    Parameters["B"] = B

    # Handle other parameters
    for param in ["freq", "step", "actionTime", "restTime", "cycles"]:
        Parameters[param] = float(df[df["Parameter"] == param]["Value"].values[0])

    # Ensure cycles is an integer
    Parameters["cycles"] = int(Parameters["cycles"])

    return Parameters


if __name__ == "__main__":
    main()
