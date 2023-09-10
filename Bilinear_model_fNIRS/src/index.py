import matplotlib.pyplot as plt
from components.BilinearModel_Neurodynamics import bilinear_model_neurodynamics_z
from components.BilinearModel_StimulusGenerator import bilinear_model_stimulus_train_generator
from components.BilinearModel_Plots import *
from components.Parameters import Parameters


def on_key(event):
    if event.key == 'escape':
        plt.close('all')


def main():
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters['freq'], Parameters['actionTime'], Parameters['restTime'], Parameters['cycles'])
    Z = bilinear_model_neurodynamics_z(
        Parameters['A'], Parameters['B'], Parameters['C'], U_stimulus, Parameters['step'])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    plot_Stimulus(U_stimulus, timestamps, fig1, ax1)
    plot_neurodynamics(Z, timestamps, fig2, ax2)

    fig1.canvas.mpl_connect('key_press_event', on_key)
    fig2.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == "__main__":
    main()
