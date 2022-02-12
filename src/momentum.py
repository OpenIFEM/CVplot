import sys
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
cv_data["Normalized time"] = (cv_data["Time"] - meta_data.timespan[0])/T_cycle


def create_momentum_frame(cv_data):
    # Copy momentum columns
    cv_momentum = cv_data[["Time", "Normalized time", "Inlet pressure force",
                           "Outlet pressure force", "VF drag", "Friction force", "Momentum change rate"]].copy()
    # influx
    cv_momentum["Momentum influx"] = cv_data["Inlet momentum flux"] - \
        cv_data["outlet momentum flux"]
    cv_momentum["Momentum efflux"] = - cv_data["Inlet momentum flux"] + \
        cv_data["outlet momentum flux"]
    # TGP
    cv_momentum["TGP force"] = -cv_data["Outlet pressure force"] + \
        cv_data["Inlet pressure force"]
    # Pressure decomposition
    cv_momentum["2P_A^+S"] = (cv_momentum["Inlet pressure force"] +
                              cv_data["Inlet volume flow"] * rho * c)
    cv_momentum["-P_DS"] = -cv_momentum["Outlet pressure force"]
    cv_momentum["2P_D^-S"] = -(cv_data["Outlet pressure force"] -
                               cv_data["Outlet volume flow"] * rho * c)
    cv_momentum["Built-up pressure force"] = cv_momentum["2P_A^+S"] - \
        cv_momentum["2P_D^-S"]
    # Drag
    cv_momentum["-VF drag"] = -cv_momentum["VF drag"]
    # Volume source
    cv_momentum["Volume source"] = - (
        cv_data["Outlet volume flow"] - cv_data["Inlet volume flow"]) * rho * c
    cv_momentum["Radiated pressure force"] = -(
        cv_data["Outlet volume flow"] + cv_data["Inlet volume flow"]) * rho * c
    # Convert the unit
    for label, content in cv_momentum.items():
        if label != "Time" and label != "Normalized time":
            cv_momentum[label] = content * to_newton
    return cv_momentum


cv_momentum = create_momentum_frame(cv_data)

# smooth
smooth_range = 100
smooth_data(cv_momentum, meta_data, smooth_range)

# momentum total change
cv_momentum["Momentum total change"] = cv_momentum["Momentum change rate"] - \
    cv_momentum["Momentum influx"]
cv_momentum["-Momentum total change"] = - cv_momentum["Momentum change rate"] + \
    cv_momentum["Momentum influx"]

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
    plt.locator_params(axis='y', nbins=6)
    fig.tick_params(direction='in', length=20,
                    width=2, top=True, right=True)
    fig.get_legend().remove()
    fig.grid()
    fig.set_xlim(normalized_timespan)
    # Use ylim in plot settings if given
    if ("ylim" in item for item in documents["pressure"]):
        ylim = next(d for i, d in enumerate(
            documents["momentum"]) if "ylim" in d)
        fig.set_ylim(ylim["ylim"])
    fig.set_xlabel("t/T", fontsize=axis_label_size)
    fig.set_ylabel("Momentum Equation Terms (N)", fontsize=axis_label_size)


def draw_open_close(fig):
    fig.set_ylim(fig.get_ylim())
    plt.plot([meta_data.open_glottis[0], meta_data.open_glottis[0]],
             fig.get_ylim(), 'r--', linewidth=4)
    plt.plot([meta_data.open_glottis[1], meta_data.open_glottis[1]],
             fig.get_ylim(), 'r--', linewidth=4)


def update_xlabels(fig):
    ylabels = [format(label, '.2f') for label in fig.get_yticks()]
    fig.set_yticklabels(ylabels)


# Plots
# Fig 11.a: Pcv, Pcv_uns, Pcv_con
# Pcv_uns is momentum change rate
# Pcv_con is momentum efflux
fig_11a = cv_momentum.plot(
    x="Normalized time", y=["Momentum total change", "Momentum change rate", "Momentum efflux"],
    style=['-', '-.', '--'], color=['k', 'r', 'b'], markevery=50, lw=5)
apply_fig_settings(fig_11a)
draw_open_close(fig_11a)
update_xlabels(fig_11a)
# Save the plot
# plt.tight_layout()
plt.savefig(meta_data.output_dir + "/11a.png", format='png')

# Fig 11.b: Fp, FpD, Ff, -Pcv
# Fp is TGP force
# Fpd is drag force
# Ff is friction force
fig_11b = cv_momentum.plot(
    x="Normalized time", y=["TGP force", "VF drag", "Friction force", "-Momentum total change"],
    style=['-', '-', '--', '-.'], color=['lightgreen', 'r', 'k', 'b'], lw=5)
apply_fig_settings(fig_11b)
draw_open_close(fig_11b)
update_xlabels(fig_11b)
# Save the plot
# plt.tight_layout()
plt.savefig(meta_data.output_dir + "/11b.png", format='png')

# Fig 11.c: Fp + FpD, Ff, -Pcv
cv_momentum["fp+fPd"] = cv_momentum["TGP force"] + cv_momentum["VF drag"]
fig_11c = cv_momentum.plot(
    x="Normalized time", y=["fp+fPd", "Friction force", "-Momentum total change"],
    style=['-', '--', '-.'], color=['lightgreen', 'k', 'b'], lw=5)
apply_fig_settings(fig_11c)
draw_open_close(fig_11c)
update_xlabels(fig_11c)
# Save the plot
plt.savefig(meta_data.output_dir + "/11c.png", format='png')
plt.show()

width_scaling = 1.25
plt.rcParams["figure.figsize"] = [width*width_scaling, height]
plt.rcParams["figure.subplot.left"] = (
    meta_data.size["left"] + (width_scaling - 1) / 2) / width_scaling
plt.rcParams["figure.subplot.right"] = (
    meta_data.size["right"] + (width_scaling - 1) / 2) / width_scaling
plt.rcParams["legend.fontsize"] = 33
# Overall in momentum equation
momentum_overall = cv_momentum.plot(
    x="Normalized time", y=["TGP force", "-VF drag", "Friction force",
                            "Momentum change rate", "Momentum efflux"],
    style=['-', '--', '-.', '-', '--'], color=['b', 'g', 'r', 'm', 'k'], lw=5)
apply_fig_settings(momentum_overall)
draw_open_close(momentum_overall)
update_xlabels(momentum_overall)
momentum_overall.legend(
    ["TGP force",
     "-VF drag",
     "Friction force",
     "Momentum change rate",
     "Momentum efflux"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', fontsize=30, ncol=1, labelspacing=2, frameon=False)
# Save the plot
plt.savefig(meta_data.output_dir + "/cv_source_overall.png", format='png')


# Plot volume source and drag soruce
volume_drag_source = cv_momentum.plot(
    x="Normalized time", y=["Volume source", "-VF drag"],
    style=['-', '--'], color=['b', 'g'], lw=5)
apply_fig_settings(volume_drag_source)
draw_open_close(volume_drag_source)
update_xlabels(volume_drag_source)
volume_drag_source.legend(
    ["Volume source",
     "-VF drag", ],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
# Save the plot
plt.savefig(meta_data.output_dir + "/cv_source_volume_drag.png", format='png')
plt.show()

# Plot TGP decomposition
tgp_decomp_a = cv_momentum.plot(
    x="Normalized time", y=["TGP force", "Inlet pressure force", "-P_DS"],
    style=['-', '--', '-.'], color=['b', 'g', 'r'], lw=5)
apply_fig_settings(tgp_decomp_a)
draw_open_close(tgp_decomp_a)
update_xlabels(tgp_decomp_a)
tgp_decomp_a.legend(
    ["TGP force",
     r"$\langle p_\mathrm{A} \rangle S_\mathrm{VT}$",
     r"$-\langle p_\mathrm{D} \rangle S_\mathrm{VT}$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
# Save the plot
plt.savefig(meta_data.output_dir +
            "/cv_source_tgp_decomposition_a.png", format='png')
plt.show()

tgp_decomp_b = cv_momentum.plot(
    x="Normalized time", y=["TGP force", "Built-up pressure force",
                            "Radiated pressure force"],
    style=['-', '--', '-.'], color=['b', 'g', 'r'], lw=5)
apply_fig_settings(tgp_decomp_b)
draw_open_close(tgp_decomp_b)
update_xlabels(tgp_decomp_b)
tgp_decomp_b.legend(
    ["TGP force",
     r"$2(\langle{p}_\mathrm{A}^+\rangle -\langle{p}_\mathrm{D}^-\rangle) S_\mathrm{VT}$",
     r"$-(\frac{\rho c}{S}\langle{Q}_\mathrm{A}\rangle+\frac{\rho c}{S}\langle{Q}_\mathrm{D}\rangle)$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
# Save the plot
plt.savefig(meta_data.output_dir +
            "/cv_source_tgp_decomposition_b.png", format='png')
plt.show()
