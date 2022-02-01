import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
to_kg = 1e-3  # Convert from g to kg

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


def create_mass_frame(cv_data):
    # Copy energy columns
    rho = 1.3e-3  # g/cm^3
    W = 1.394  # width in cm
    cv_mass = cv_data[["Time", "Normalized time"]].copy()
    cv_mass["Inlet mass flow"] = cv_data["Inlet volume flow"] * rho * W
    cv_mass["Outlet mass flow"] = cv_data["Outlet volume flow"] * rho * W
    cv_mass["Gap mass flow"] = cv_data["Gap volume flow"] * rho * W
    cv_mass["Mass change rate"] = (
        cv_mass["Inlet mass flow"] - cv_mass["Outlet mass flow"])
    cv_mass["Mass from VF"] = cv_data["VF volume"] * rho
    # Compute mass rate from VF
    mass_VF = cv_mass["Mass from VF"].to_numpy()
    mass_rate_VF = np.zeros(len(mass_VF))
    dt = 5e-7
    for i in range(len(mass_VF)):
        if i != 0:
            mass_rate_VF[i] = -(mass_VF[i] - mass_VF[i-1])/dt

    cv_mass["Mass change rate from VF"] = mass_rate_VF
    # Convert the unit
    for label, content in cv_mass.items():
        if label != "Time" and label != "Normalized time":
            cv_mass[label] = content * to_kg
    return cv_mass


cv_mass = create_mass_frame(cv_data)

# smooth
smooth_range = 100
smooth_data(cv_mass, meta_data, smooth_range)

# Compute maximum gap mass flow
# max_gap_mass_flow = max(cv_mass["Gap mass flow"].to_numpy())
# Get the mass flow when the Bernoulli start having residual
# res_T = 0.791
# max_res_T = 0.856
# gap_mass_flow_nonzero_res = np.interp(res_T,
#                                       cv_mass["Normalized time"].to_numpy(), cv_mass["Gap mass flow"].to_numpy())
# print(f"gap mass flow is {gap_mass_flow_nonzero_res / max_gap_mass_flow}")
# gap_mass_flow_max_res = np.interp(max_res_T,
#                                   cv_mass["Normalized time"].to_numpy(), cv_mass["Gap mass flow"].to_numpy())
# print(
#     f"gap mass flow at max res is {gap_mass_flow_max_res / max_gap_mass_flow}")

# Figure properties
height = 938/80
width = 1266/80
label_size = 36
plt.rcParams["figure.figsize"] = [width, height]
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
    fig.set_ylabel("Mass Flowrate (kg/s)", fontsize=axis_label_size)


def draw_open_close(fig):
    fig.set_ylim(fig.get_ylim())
    plt.plot([meta_data.open_glottis[0], meta_data.open_glottis[0]],
             fig.get_ylim(), 'r--', linewidth=4)
    plt.plot([meta_data.open_glottis[1], meta_data.open_glottis[1]],
             fig.get_ylim(), 'r--', linewidth=4)
    # plt.plot(fig.get_xlim(), [gap_mass_flow_nonzero_res,
    #                           gap_mass_flow_nonzero_res], 'k--', linewidth=2)
    # plt.plot(fig.get_xlim(), [gap_mass_flow_max_res,
    #                           gap_mass_flow_max_res], 'k--', linewidth=2)


# Plots
# Fig 12a: W_Fp, W_structure, W_f, KE_CV,uns, KE_CV,con, W_c, W_v
fig_10 = cv_mass.plot(
    x="Normalized time",
    y=["Inlet mass flow", "Outlet mass flow",
        "Mass change rate"],
    style=['--', '-.', '-'],
    color=['b', 'r', 'k'], markevery=20, lw=5)
apply_fig_settings(fig_10)
draw_open_close(fig_10)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/10.png", format='png')

fig_8b = cv_mass.plot(
    x="Normalized time",
    y=["Gap mass flow"],
    style=['-'],
    color=['g'], markevery=20, lw=5)
apply_fig_settings(fig_8b)
draw_open_close(fig_8b)


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.2f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


fig_8b.yaxis.set_major_formatter(OOMFormatter(order=-3))
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/glottal_flow.png", format='png')
plt.show()
