import sys
from zipapp import create_archive
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from meta_data import CVMetaData

from smooth import *

# Global variables
S = 1.397  # Cross section area (height in 2D)
c = 34000  # Sound speed
rho = 1.3e-3  # Air density
to_newton = 1e-5  # Convert from dyne to Newton
to_pa = 1e-1  # Convert from dyne/cm^2 to Pascal
to_watt = 1e-7  # Convert from erg/s to Watt


def create_pressure_frame(cv_data):
    cv_pressure = cv_data[["Time", "Normalized time"]].copy()
    # TGP
    cv_pressure["TGP"] = (cv_data["Inlet pressure force"] -
                          cv_data["Outlet pressure force"]) / S
    # Pressure decomposition
    cv_pressure["P_A"] = cv_data["Inlet pressure force"] / S
    cv_pressure["P_D"] = cv_data["Outlet pressure force"] / S
    cv_pressure["P_A^+"] = (cv_pressure["P_A"] +
                            cv_data["Inlet volume flow"] * rho * c / S)/2
    cv_pressure["P_D^-"] = (cv_pressure["P_D"] -
                            cv_data["Outlet volume flow"] * rho * c / S)/2
    cv_pressure["P_A^-"] = cv_pressure["P_A"] - cv_pressure["P_A^+"]
    cv_pressure["P_D^+"] = cv_pressure["P_D"] - cv_pressure["P_D^-"]
    cv_pressure["Built-up pressure"] = 2 * \
        (cv_pressure["P_A^+"] - cv_pressure["P_D^-"])
    cv_pressure["Entrance built-up pressure"] = 2 * cv_pressure["P_A^+"]
    cv_pressure["Exit built-up pressure"] = 2 * cv_pressure["P_D^-"]
    cv_pressure["Radiated pressure"] = (
        cv_data["Outlet volume flow"] + cv_data["Inlet volume flow"]) * rho * c / S
    cv_pressure["Entrance radiated pressure"] = cv_data["Inlet volume flow"] * rho * c / S
    cv_pressure["Exit radiated pressure"] = cv_data["Outlet volume flow"] * rho * c / S
    # Convert the unit
    for label, content in cv_pressure.items():
        if label != "Time" and label != "Normalized time":
            cv_pressure[label] = content * to_pa
    return cv_pressure


def main():
    meta_data = CVMetaData(sys.argv)
    documents = meta_data.documents
    # Read CV data file
    cv_data = pd.read_csv(meta_data.filename, header=0)

    # Time
    time = cv_data["Time"]
    # Time span
    normalized_timespan = [0.0, meta_data.n_period]
    time_to_plot = meta_data.timespan[1] - meta_data.timespan[0]
    T_cycle = time_to_plot/meta_data.n_period
    cv_data["Normalized time"] = (
        cv_data["Time"] - meta_data.timespan[0])/T_cycle

    cv_pressure = create_pressure_frame(cv_data)

    # smooth
    smooth_range = 100
    smooth_data(cv_pressure, meta_data, smooth_range)

    # Figure properties
    height = 938/80
    width = 1266/80
    label_size = 36
    plt.rcParams["figure.figsize"] = [width, height]
    plt.rcParams["legend.fontsize"] = label_size
    plt.rcParams["xtick.labelsize"] = label_size
    plt.rcParams["ytick.labelsize"] = label_size

    def apply_fig_settings(fig):
        axis_label_size = 36
        plt.locator_params(axis='y', nbins=8)
        fig.tick_params(direction='in', length=20,
                        width=2, top=True, right=True)
        fig.get_legend().remove()
        fig.grid()
        fig.set_xlim(normalized_timespan)
        fig.set_xlabel("t/T", fontsize=axis_label_size)
        fig.set_ylabel("Pressure (Pa)", fontsize=axis_label_size)

    def draw_open_close(fig):
        fig.set_ylim(fig.get_ylim())
        # Use ylim in plot settings if given
        if ("ylim" in item for item in documents["pressure"]):
            ylim = next(d for i, d in enumerate(
                documents["pressure"]) if "ylim" in d)
            fig.set_ylim(ylim["ylim"])
        plt.plot([meta_data.open_glottis[0], meta_data.open_glottis[0]],
                 fig.get_ylim(), 'r--', linewidth=4)
        plt.plot([meta_data.open_glottis[1], meta_data.open_glottis[1]],
                 fig.get_ylim(), 'r--', linewidth=4)

    def update_xlabels(fig):
        ylabels = [format(label, '.0f') for label in fig.get_yticks()]
        fig.set_yticklabels(ylabels)

    # Plots
    pressure_waveform = cv_pressure.plot(
        x="Normalized time", y=["P_A", "P_D"],
        style=['-', '--'], color=['b', 'g'], markevery=50, lw=5)
    apply_fig_settings(pressure_waveform)
    draw_open_close(pressure_waveform)
    update_xlabels(pressure_waveform)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir +
                "/cv_pressure_waveform.png", format='png')
    plt.show()

    # Entrance pressure decomposition
    entrance_decomp_a = cv_pressure.plot(
        x="Normalized time", y=["P_A", "P_A^+", "P_A^-"],
        style=['-', '--', '-.'], color=['b', 'g', 'r'], markevery=50, lw=5)
    apply_fig_settings(entrance_decomp_a)
    draw_open_close(entrance_decomp_a)
    update_xlabels(entrance_decomp_a)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir +
                "/cv_entrance_pressure_decomposition_a.png", format='png')
    plt.show()

    # Entrance pressure decomposition 2
    entrance_decomp_b = cv_pressure.plot(
        x="Normalized time",
        y=["P_A", "Entrance built-up pressure", "Entrance radiated pressure"],
        style=['-', '--', '-.'], color=['b', 'g', 'r'], markevery=50, lw=5)
    apply_fig_settings(entrance_decomp_b)
    draw_open_close(entrance_decomp_b)
    update_xlabels(entrance_decomp_b)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir +
                "/cv_entrance_pressure_decomposition_b.png", format='png')
    plt.show()

    # Exit pressure decomposition
    exit_decomp_a = cv_pressure.plot(
        x="Normalized time", y=["P_D", "P_D^+", "P_D^-"],
        style=['-', '--', '-.'], color=['b', 'g', 'r'], markevery=50, lw=5)
    apply_fig_settings(exit_decomp_a)
    draw_open_close(exit_decomp_a)
    update_xlabels(exit_decomp_a)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir +
                "/cv_exit_pressure_decomposition_a.png", format='png')
    plt.show()

    # Exit pressure decomposition 2
    exit_decomp_b = cv_pressure.plot(
        x="Normalized time",
        y=["P_D", "Exit built-up pressure", "Exit radiated pressure"],
        style=['-', '--', '-.'], color=['b', 'g', 'r'], markevery=50, lw=5)
    apply_fig_settings(exit_decomp_b)
    draw_open_close(exit_decomp_b)
    update_xlabels(exit_decomp_b)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir +
                "/cv_exit_pressure_decomposition_b.png", format='png')
    plt.show()


if __name__ == "__main__":
    main()
