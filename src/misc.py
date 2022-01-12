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
to_mps = 1e-2  # Convert from cm/s to m/s
to_pa = 1e-1  # Convert from dyne to Pa

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

on_fly_data = cv_data[["Time", "Normalized time",
                       "Max velocity", "Probed pressure"]].copy()
# Convert units
on_fly_data["Max velocity"] = on_fly_data["Max velocity"] * to_mps
on_fly_data["Probed pressure"] = on_fly_data["Probed pressure"] * to_pa

# Smooth
smooth_start_index = 0
smooth_end_index = 0
smooth_range = 100
for index in range(len(cv_data["Normalized time"])):
    if cv_data["Normalized time"][index] < 0:
        smooth_start_index = index
    if cv_data["Normalized time"][index] < meta_data.n_period:
        smooth_end_index = index
smooth_start_index -= smooth_range
smooth_end_index += smooth_range
for label, content in on_fly_data.items():
    if label != "Time" and label != "Normalized time":
        direct_smooth(on_fly_data, label, label, smooth_range, [
                      smooth_start_index, smooth_end_index])

# Figure properties
height = 938/80
width = 1266/80
label_size = 36
axis_label_size = 36
plt.rcParams["figure.figsize"] = [width, height]
plt.rcParams["xtick.labelsize"] = label_size
plt.rcParams["ytick.labelsize"] = label_size


def apply_fig_settings(fig):
    plt.locator_params(axis='y', nbins=8)
    fig.tick_params(direction='in', length=20,
                    width=2, top=True, right=True)
    fig.get_legend().remove()
    fig.grid()
    fig.set_xlim(normalized_timespan)


def draw_open_close(fig):
    fig.set_ylim(fig.get_ylim())
    plt.plot([meta_data.open_glottis[0], meta_data.open_glottis[0]],
             fig.get_ylim(), 'r--', linewidth=4)
    plt.plot([meta_data.open_glottis[1], meta_data.open_glottis[1]],
             fig.get_ylim(), 'r--', linewidth=4)


# FFT for pressure
def do_psd(p, t):
    # Hann window
    p = np.multiply(np.hanning(len(p)), p)
    Ts = t[1]-t[0]
    # Use the next2power as N
    N = int(2**np.ceil(np.log(p.shape[0])/np.log(2))) * 10
    Nf = N // 2
    xf = np.fft.fftfreq(N, d=Ts)
    yf = 2.0 * np.fft.fft(p, N) ** 2 * Ts / N * 10
    return xf[:Nf], yf[:Nf]


# Plots
max_vel = on_fly_data.plot(
    x="Normalized time",
    y=["Max velocity"],
    style=['-'],
    color=['b'], markevery=20, lw=5)
apply_fig_settings(max_vel)
draw_open_close(max_vel)
max_vel.set_xlabel("t/T", fontsize=axis_label_size)
max_vel.set_ylabel("Maximum velocity (m/s)", fontsize=axis_label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/max_vel.png", format='png')

pressure_time = on_fly_data.plot(
    x="Normalized time",
    y=["Probed pressure"],
    style=['-'],
    color=['b'], markevery=20, lw=5)
apply_fig_settings(pressure_time)
draw_open_close(pressure_time)
pressure_time.set_xlabel("t/T", fontsize=axis_label_size)
pressure_time.set_ylabel("Mouth pressure (Pa)", fontsize=axis_label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/pressure_time_domaion.png", format='png')
plt.show()
