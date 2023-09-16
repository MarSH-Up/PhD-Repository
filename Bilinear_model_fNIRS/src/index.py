import matplotlib.pyplot as plt
from components.BilinearModel_Hemodynamics import hemodynamics_system
from components.BilinearModel_Plots import *
from components.BilinearModel_StimulusGenerator import (
    bilinear_model_stimulus_train_generator,
)
from components.Models import *
from components.Parameters import Parameters
from scipy.integrate import odeint


def on_key(event):
    if event.key == "escape":
        plt.close("all")


def fNIRS_Process():
    U_stimulus, timestamps = bilinear_model_stimulus_train_generator(
        Parameters["freq"],
        Parameters["actionTime"],
        Parameters["restTime"],
        Parameters["cycles"],
        Parameters["A"].shape[0],
    )

    Z0 = np.zeros([Parameters["A"].shape[0]])
    Z = odeint(
        Neurodynamics_Z,
        Z0,
        t=timestamps,
        args=(Parameters["A"], Parameters["B"], Parameters["C"], U_stimulus),
    )

    # Initial conditions
    # Initial conditions for 2 regions
    Sj0 = np.array([0, 0])
    fjin0 = np.array([1, 1])
    Vj0 = np.array([1, 1])
    Pj0 = np.array([1, 1])
    qj0 = np.array([1, 1])

    # Combine initial conditions
    Y0 = np.concatenate([Sj0, fjin0, Vj0, Pj0, qj0])

    # Parameters from P_SD and H
    P_SD = np.array(
        [[0.0775, -0.0087], [-0.1066, 0.0299], [0.0440, -0.0129], [0.8043, -0.7577]]
    )

    result = odeint(
        hemodynamics_system,
        Y0,
        timestamps,
        args=(Z, U_stimulus, P_SD, Parameters["A"], Parameters["step"]),
    )

    Sj, fjin, Vj, pj, qj = result.T
    print(qj, pj)

    return U_stimulus, timestamps, Z


def main():
    U_stimulus, timestamps, Z = fNIRS_Process()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    plot_Stimulus(U_stimulus, timestamps, fig1, ax1)
    plot_neurodynamics(Z, timestamps, fig2, ax2)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()


# https://towardsdatascience.com/ordinal-differential-equation-ode-in-python-8dc1de21323b
