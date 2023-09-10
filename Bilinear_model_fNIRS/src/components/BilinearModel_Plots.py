import numpy as np
import matplotlib.pyplot as plt


def plot_neurodynamics(Z, timestamps, fig, ax):
    for i in range(Z.shape[0]):
        ax.plot(timestamps, Z[i, :], label=f"Motor Execution {i+1}")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurodynamic Value')
    ax.set_title('Neurodynamics vs Time')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.show()

def plot_Stimulus(U_stimulus, timestamps, fig, ax):
    if U_stimulus.ndim == 2:
        for i in range(U_stimulus.shape[0]):
            ax.plot(timestamps, U_stimulus[i])
    else:
        ax.plot(timestamps, U_stimulus)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stimulus Value')
    ax.set_title('Stimulus vs Time')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.show()
