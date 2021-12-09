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
cv_data["Normalized time"] = (cv_data["Time"] - meta_data.timespan[0])/T_cycle


def create_energy_frame(cv_data):
    # Copy energy columns
    cv_energy = cv_data[["Time", "Normalized time"]].copy()
    # Rate KE
    if ("Rate KE" in documents["energy"]):
        cv_energy["Rate KE"] = -cv_data["Rate KE direct"]
    # Efflux
    if ("-KE efflux" in documents["energy"]):
        cv_energy["-KE efflux"] = cv_data["Inlet KE flux"] - \
            cv_data["Outlet KE flux"]
    if ("Rate dissipation" in documents["energy"]):
        cv_energy["Rate dissipation"] = -cv_data["Rate dissipation"]
    if ("Rate compression work" in documents["energy"]):
        cv_energy["Rate compression work"] = cv_data["Rate compression work"]
    if ("Rate friction work" in documents["energy"]):
        cv_energy["Rate friction work"] = -cv_data["Rate friction work"]
    if ("Pressure drive" in documents["energy"]):
        cv_energy["Pressure drive"] = cv_data["Inlet pressure work"] - \
            cv_data["Outlet pressure work"]
    # VF work
    if ("Rate VF work" in documents["energy"]):
        cv_energy["-Convective KE"] = -cv_data["Convective KE"]
        cv_energy["Rate VF work"] = -cv_data["Pressure convection"] - cv_energy["Pressure drive"]\
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
    cv_energy["Residual"] = get_or_none("Pressure drive") + \
        get_or_none("-KE efflux") + get_or_none("Rate KE") + \
        get_or_none("Rate VF work") + get_or_none("Rate compression work") + \
        get_or_none("Rate dissipation") + get_or_none("Rate friction work") - \
        get_or_none("Stabilization") +\
        get_or_none("Rate turbulent momentum transfer")

    # Convert the unit
    for label, content in cv_energy.items():
        if label != "Time" and label != "Normalized time":
            cv_energy[label] = content * to_watt
    return cv_energy


cv_energy = create_energy_frame(cv_data)

# smooth
for label, content in cv_energy.items():
    if label != "Time" and label != "Normalized time":
        direct_smooth(cv_energy, label, label, 100)

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
    if "legend" in documents and documents["legend"] == True:
        print(documents["legend"])
        fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        fig.get_legend().remove()
    fig.grid()
    fig.set_xlim(normalized_timespan)
    fig.set_xlabel("t/T", fontsize=axis_label_size)
    fig.set_ylabel("Energy Equation Terms (Watts)", fontsize=axis_label_size)


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
for (item, line, color) in zip(["Residual", "Pressure drive", "Rate VF work", "Rate friction work",
                                "Rate KE", "-KE efflux", "Rate compression work",
                                "Rate dissipation", "Stabilization",
                                "Rate turbulent momentum transfer"],
                               ['-', '-', '-', '--', '-',
                                   '-.', '--', '-.', '-', '-'],
                               ['g', 'lightgreen', 'r', 'k', 'b', 'b', 'b', 'm', 'y', 'm']):
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
    if line.get_label() == "Pressure drive":
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

# Output the budget
buget_table = [("Energy Term", "Percentage")]
integrated_works = {}

one_cycle_data = cv_energy[(cv_energy["Normalized time"] >= 0) & (
    cv_energy["Normalized time"] <= 1)]

for term in one_cycle_data.items():
    term_name = term[0]
    if term_name == "Time" or term_name == "Normalized time":
        continue
    integrated_works[term_name] = 0.0
    for entry in one_cycle_data[term_name]:
        integrated_works[term_name] += entry * 5e-7

for term in integrated_works.items():
    percentage = f"{term[1]/integrated_works['Pressure drive'] * 100:.1f}%"
    buget_table.append((term[0], percentage))

print(tabulate(buget_table, headers='firstrow', tablefmt='fancy_grid'))
