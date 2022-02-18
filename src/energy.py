import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from tabulate import tabulate
from meta_data import CVMetaData

from smooth import *

# Global variables
S = 1.397  # Cross section area (height in 2D)
c = 34000  # Sound speed
rho = 1.3e-3  # Air density
to_newton = 1e-5  # Convert from dyne to Newton
to_pa = 1e-1  # Convert from dyne/cm^2 to Pascal
to_watt = 1e-7  # Convert from erg/s to Watt


def create_energy_frame(cv_data, documents):
    # Copy energy columns
    cv_energy = cv_data[["Time", "Normalized time"]].copy()
    # Rate KE
    if ("Rate KE" in documents["energy"]):
        cv_energy["Rate KE"] = -cv_data["Rate KE direct"]
    # Efflux
    if ("-KE efflux" in documents["energy"]):
        cv_energy["-KE efflux"] = cv_data["Inlet KE flux"] - \
            cv_data["Outlet KE flux"]
        cv_energy["-Convective KE"] = -cv_data["Convective KE"]
        cv_energy["Rate KE loss due to compression"] = cv_energy["-Convective KE"] - \
            cv_energy["-KE efflux"]
    if ("Rate dissipation" in documents["energy"]):
        cv_energy["Rate dissipation"] = -cv_data["Rate dissipation"]
    if ("Rate compression work" in documents["energy"]):
        cv_energy["Rate compression work"] = cv_data["Rate compression work"]
    if ("Rate friction work" in documents["energy"]):
        cv_energy["Rate friction work"] = -cv_data["Rate friction work"]
    if ("Driving pressure work" in documents["energy"]):
        cv_energy["Inlet pressure work"] = cv_data["Inlet pressure work"]
        cv_energy["Outlet pressure work"] = cv_data["Outlet pressure work"]
        cv_energy["Driving pressure work"] = cv_data["Inlet pressure work"] - \
            cv_data["Outlet pressure work"]
        cv_energy["Acoustic output"] = - \
            cv_data["Outlet volume flow"]**2 * rho * c / S
        cv_energy["Acoustic loss"] = - \
            cv_data["Inlet volume flow"]**2 * rho * c / S
        cv_energy["Pressure input"] = cv_data["Inlet pressure work"] - cv_energy["Acoustic loss"] - \
            (cv_data["Outlet pressure work"] + cv_energy["Acoustic output"])
        cv_energy["2P_A^+Q_A"] = (cv_data["Inlet pressure work"] +
                                  cv_data["Inlet volume flow"]**2 * rho * c / S)
        cv_energy["2P_D^-Q_D"] = (cv_data["Outlet pressure work"] -
                                  cv_data["Outlet volume flow"]**2 * rho * c / S)
        cv_energy["Net acoustic power flow"] = cv_energy["Acoustic output"] + \
            cv_energy["Acoustic loss"]
    # VF work
    if ("Rate VF work" in documents["energy"]):
        cv_energy["Rate VF work"] = -cv_data["Pressure convection"] - cv_energy["Driving pressure work"]\
            - cv_data["Rate compression work"]
    # Stabilization
    if ("Stabilization" in documents["energy"]):
        cv_energy["Stabilization"] = cv_data["Rate stabilization"]
    # Turbulence
    if ("Rate turbulent momentum transfer" in documents["energy"]):
        cv_energy["Rate turbulent momentum transfer"] = - cv_data["Rate turbulence work"]\
            - cv_data["Rate turbulence efflux"]

    def get_or_none(name: str):
        try:
            return cv_energy[name]
        except KeyError:
            return 0
    # Residual
    cv_energy["Residual"] = get_or_none("Driving pressure work") + \
        get_or_none("-KE efflux") + get_or_none("Rate KE") + \
        get_or_none("Rate VF work") + get_or_none("Rate compression work") + \
        get_or_none("Rate dissipation") + get_or_none("Rate friction work") - \
        get_or_none("Stabilization") +\
        get_or_none("Rate turbulent momentum transfer") +\
        get_or_none("Rate KE loss due to compression")

    # Convert the unit
    for label, content in cv_energy.items():
        if label != "Time" and label != "Normalized time":
            cv_energy[label] = content * to_watt
    return cv_energy


def compute_budget(cv_energy: pd.DataFrame, meta_data: CVMetaData):
    terms_to_compute = {"Rate KE", "-KE efflux", "Rate KE loss due to compression", "Rate dissipation",
                        "Rate compression work", "Rate friction work", "Driving pressure work",
                        "Acoustic output", "Acoustic loss", "Pressure input", "Net acoustic power flow",
                        "Rate VF work", "Rate turbulent momentum transfer", "Residual"}
    budget_dict = {}
    integrated_works = {}

    n_cycle_data = cv_energy[(cv_energy["Normalized time"] >= 0) & (
        cv_energy["Normalized time"] <= meta_data.n_period)]

    for term in n_cycle_data.items():
        term_name = term[0]
        if term_name not in terms_to_compute:
            continue
        integrated_works[term_name] = 0.0
        for entry in n_cycle_data[term_name]:
            integrated_works[term_name] += entry * 5e-7

    for term in integrated_works.items():
        percentage = term[1] / integrated_works["Pressure input"]
        budget_dict[term[0]] = percentage

    return budget_dict


def main():
    meta_data = CVMetaData(sys.argv)
    documents = meta_data.documents
    # Read CV data file
    cv_data = pd.read_csv(meta_data.filename, header=0)

    # Time
    time = cv_data["Time"]
    # Time span
    normalized_timespan = [0, meta_data.n_period]
    time_to_plot = meta_data.timespan[1] - meta_data.timespan[0]
    T_cycle = time_to_plot/meta_data.n_period
    cv_data["Normalized time"] = (
        cv_data["Time"] - meta_data.timespan[0])/T_cycle

    cv_energy = create_energy_frame(cv_data, documents)

    # smooth
    smooth_range = 100
    smooth_data(cv_energy, meta_data, smooth_range)

    # Figure properties
    height = meta_data.size["height"]
    width = meta_data.size["width"]
    label_size = 36
    plt.rcParams["figure.figsize"] = [width, height]
    plt.rcParams["legend.fontsize"] = label_size
    plt.rcParams["xtick.labelsize"] = label_size
    plt.rcParams["ytick.labelsize"] = label_size
    plt.rcParams["figure.subplot.left"] = meta_data.size["left"]
    plt.rcParams["figure.subplot.right"] = meta_data.size["right"]

    def apply_fig_settings(fig):
        axis_label_size = 36
        plt.locator_params(axis='y', nbins=8)
        fig.tick_params(direction='in', length=20,
                        width=2, top=True, right=True)
        if "legend" in documents and documents["legend"] == True:
            print(documents["legend"])
            fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            fig.get_legend().remove()
        fig.grid()
        fig.set_xlim(normalized_timespan)
        fig.set_xlabel("t/T", fontsize=axis_label_size)
        fig.set_ylabel("Energy Equation Terms (Watts)",
                       fontsize=axis_label_size)

    def draw_open_close(fig):
        fig.set_ylim(fig.get_ylim())
        # Use ylim in plot settings if given
        if ("ylim" in item for item in documents["energy"]):
            ylim = next(d for i, d in enumerate(
                documents["energy"]) if "ylim" in d)
            fig.set_ylim(ylim["ylim"])
        plt.plot([meta_data.open_glottis[0], meta_data.open_glottis[0]],
                 fig.get_ylim(), 'r--', linewidth=4)
        plt.plot([meta_data.open_glottis[1], meta_data.open_glottis[1]],
                 fig.get_ylim(), 'r--', linewidth=4)

    def update_xlabels(fig):
        ylabels = [format(label, '.2f') for label in fig.get_yticks()]
        fig.set_yticklabels(ylabels)

    # Plots
    fig_12a_names, fig_12a_styles, fig_12a_colors = [], [], []
    for (item, line, color) in zip(["Residual", "Driving pressure work", "Rate VF work", "Rate friction work",
                                    "Rate KE", "-KE efflux", "Rate compression work",
                                    "Rate dissipation", "Stabilization",
                                    "Rate turbulent momentum transfer", "Rate KE loss due to compression"],
                                   ['-', '-', '-', '--', '-',
                                    '-.', '--', '-.', '-', '-', '-.'],
                                   ['g', 'lightgreen', 'r', 'k', 'b', 'b', 'b', 'm', 'y', 'm', 'y']):
        if item in cv_energy.keys():
            fig_12a_names.append(item)
            fig_12a_styles.append(line)
            fig_12a_colors.append(color)

    fig_12a = cv_energy.plot(
        x="Normalized time",
        y=fig_12a_names,
        style=fig_12a_styles,
        color=fig_12a_colors, markevery=20, lw=5)
    # Thicken the input
    for line in fig_12a.get_lines():
        if line.get_label() == "Driving pressure work":
            line.set_linewidth(10)

    apply_fig_settings(fig_12a)
    draw_open_close(fig_12a)
    update_xlabels(fig_12a)
    plt.tight_layout()
    plt.savefig(meta_data.output_dir + "/12a.png", format='png')

    # Fig 12b: W_f, W_c, W_v
    fig_12b_names, fig_12b_styles, fig_12b_colors = [], [], []
    fig_12b = cv_energy.plot(
        x="Normalized time",
        y=["Rate friction work", "Rate compression work",  "Rate dissipation"],
        style=['--', '--', '-.'], color=['k', 'b', 'm'], markevery=20, lw=5)
    apply_fig_settings(fig_12b)
    draw_open_close(fig_12b)
    update_xlabels(fig_12b)
    # Save the plot
    plt.tight_layout()
    plt.savefig(meta_data.output_dir + "/12b.png", format='png')
    plt.show()

    # Figure for decomposed pressure
    width_scaling = 1.25
    plt.rcParams["figure.figsize"] = [width*width_scaling, height]
    plt.rcParams["figure.subplot.left"] = (
        meta_data.size["left"] + (width_scaling - 1) / 2) / width_scaling
    plt.rcParams["figure.subplot.right"] = (
        meta_data.size["right"] + (width_scaling - 1) / 2) / width_scaling
    plt.rcParams["legend.fontsize"] = 33
    # Net input, acoustic loss/output
    input_decomp_plot = cv_energy.plot(
        x="Normalized time", y=["Driving pressure work", "Pressure input", "Net acoustic power flow"],
        style=["b-", "g--", "r-."])
    apply_fig_settings(input_decomp_plot)
    draw_open_close(input_decomp_plot)
    update_xlabels(input_decomp_plot)
    for line in input_decomp_plot.get_lines():
        if line.get_label() == "Pressure input":
            line.set_linewidth(6)
        else:
            line.set_linewidth(5)
    input_decomp_plot.legend(
        [r"$\langle{p}_\mathrm{A}\rangle\langle{Q}_\mathrm{A}\rangle-\langle{p}_\mathrm{D}\rangle\langle{Q}_\mathrm{D}\rangle$",
         r"$2(\langle{p}_\mathrm{A}^+\rangle\langle{Q}_\mathrm{A}\rangle-\langle{p}_\mathrm{D}^-\rangle\langle{Q}_\mathrm{D}\rangle)$",
         r"$-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{A}\rangle^2-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{D}\rangle^2$"],
        bbox_to_anchor=(1.0, 0.5),
        loc='center left', ncol=1, labelspacing=2, frameon=False)
    # Save the plot
    plt.savefig(meta_data.output_dir +
                "/cv_energy_pressure_input_decomposition.png", format='png')
    plt.show()

    # Entrance decomposition
    entrance_decomp_plot = cv_energy.plot(
        x="Normalized time", y=["Inlet pressure work", "2P_A^+Q_A", "Acoustic loss"],
        style=["b-", "g--", "r-."])
    apply_fig_settings(entrance_decomp_plot)
    draw_open_close(entrance_decomp_plot)
    update_xlabels(entrance_decomp_plot)
    for line in entrance_decomp_plot.get_lines():
        if line.get_label() == "Inlet pressure work":
            line.set_linewidth(6)
        else:
            line.set_linewidth(5)
    entrance_decomp_plot.legend(
        [r"$\langle{p}_\mathrm{A}\rangle\langle{Q}_\mathrm{A}\rangle$",
         r"$2(\langle{p}_\mathrm{A}^+\rangle\langle{Q}_\mathrm{A}\rangle$",
         r"$-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{A}\rangle^2$"],
        bbox_to_anchor=(1.0, 0.5),
        loc='center left', ncol=1, labelspacing=2, frameon=False)
    # Save the plot
    plt.savefig(meta_data.output_dir +
                "/cv_energy_entrance_pressure_decomposition.png", format='png')
    plt.show()

    # Exit decomposition
    exit_decomp_plot = cv_energy.plot(
        x="Normalized time", y=["Outlet pressure work", "2P_D^-Q_D", "Acoustic output"],
        style=["b-", "g--", "r-."])
    apply_fig_settings(exit_decomp_plot)
    draw_open_close(exit_decomp_plot)
    update_xlabels(exit_decomp_plot)
    for line in exit_decomp_plot.get_lines():
        if line.get_label() == "Outlet pressure work":
            line.set_linewidth(6)
        else:
            line.set_linewidth(5)
    exit_decomp_plot.legend(
        [r"$\langle{p}_\mathrm{D}\rangle\langle{Q}_\mathrm{D}\rangle$",
         r"$2(\langle{p}_\mathrm{D}^-\rangle\langle{Q}_\mathrm{D}\rangle$",
         r"$-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{D}\rangle^2$"],
        bbox_to_anchor=(1.0, 0.5),
        loc='center left', ncol=1, labelspacing=2, frameon=False)
    # Save the plot
    plt.savefig(meta_data.output_dir +
                "/cv_energy_exit_pressure_decomposition.png", format='png')
    plt.show()

    # All terms with pressure decomposition
    energy_plot = cv_energy.plot(
        x="Normalized time", y=["Pressure input", "Acoustic loss", "Acoustic output",
                                "Rate VF work", "Rate KE", "-KE efflux",  "Rate KE loss due to compression",
                                "Rate dissipation", "Rate compression work", "Rate friction work",
                                "Rate turbulent momentum transfer"],
        style=["b", "r", "b--", "b", "g", "r--", "m--", "r-.", "g--", "m", "k"], markevery=20)
    for line in energy_plot.get_lines():
        if line.get_label() == "Pressure input":
            line.set_linewidth(6)
        else:
            line.set_linewidth(3)
    apply_fig_settings(energy_plot)
    draw_open_close(energy_plot)
    update_xlabels(energy_plot)
    energy_legend = [r"$2(\langle{p}_\mathrm{A}^+\rangle\langle{Q}_\mathrm{A}\rangle-\langle{p}_\mathrm{D}^-\rangle\langle{Q}_\mathrm{D}\rangle)$",
                     r"$-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{A}\rangle^2$", r"$-\frac{\rho c}{S_\mathrm{VT}}\langle{Q}_\mathrm{D}\rangle^2$",
                     r"$-\dot{W}_\mathrm{VF}$", r"$-\dot{KE}_\mathrm{V}$", r"$-\dot{KE}_\mathrm{S}$", r"$-\dot{KE}_{VC}$",
                     r"$-\dot{W}_\mathrm{\nu}$", r"$\dot{PE}$", r"$-\dot{W}_\mathrm{f}$", r"$-\dot{W}_\mathrm{t}$"]
    energy_plot.legend(energy_legend, bbox_to_anchor=(1.0, 0.5),
                       loc="center left", ncol=1, labelspacing=0.55, frameon=False)
    # Save the plot
    plt.savefig(meta_data.output_dir +
                "/cv_energy_overall.png", format='png')
    plt.show()

    # Output the budget
    budget_dict = compute_budget(cv_energy, meta_data)
    budget_table = [("Energy Term", "Percentage")]
    for key, item in budget_dict.items():
        percentage = f"{100*item:.2f}%"
        budget_table.append((key, percentage))
    print(tabulate(budget_table, headers='firstrow', tablefmt='fancy_grid'))


if __name__ == "__main__":
    main()
