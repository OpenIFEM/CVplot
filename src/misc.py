import sys
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from meta_data import CVMetaData

from smooth import *

# Global variables
S = 1.397  # Cross section area (height in 2D)
c = 34000  # Sound speed
rho = 1.3e-3  # Air density
to_mps = 1e-2  # Convert from cm/s to m/s
to_pa = 1e-1  # Convert from dyne to Pa
to_mm = 1e1  # Convert from cm to mm

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

# Read gap data (if exist)
gap_data = pd.read_csv(meta_data.working_dir + "/gap.csv", header=0)
gap_data["Gap"] = gap_data["Gap"] * to_mm
gap_data["Normalized time"] = (
    gap_data["Time"] - meta_data.timespan[0])/T_cycle


# FFT for pressure
def do_psd(p, t):
    # Hann window
    p = np.multiply(np.hanning(len(p)), p)
    Ts = t[1]-t[0]
    # Use the next2power as N
    N = int(2**np.ceil(np.log(p.shape[0])/np.log(2))) * 10
    Nf = N // 2
    xf = np.fft.fftfreq(N, d=Ts)
    yf = 2.0 * (np.fft.fft(p, N))**2 * Ts / N * 10
    return xf[:Nf], yf[:Nf]


freq, power = do_psd(on_fly_data["Probed pressure"], on_fly_data["Time"])
freq_gap, power_gap = do_psd(gap_data["Gap"], gap_data["Time"])

# Create resonance for pressure (broad peaks)
resonance = np.zeros(len(power))
resonance = np.log(np.abs(power))
resonance = savgol_filter(resonance, 701, 3)
resonance = savgol_filter(resonance, 1001, 3)
resonance = savgol_filter(resonance, 1401, 3)
resonance = np.exp(resonance)

# Smooth
smooth_range = 100
smooth_data(on_fly_data, meta_data, smooth_range)

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

gap_time = gap_data.plot(
    x="Normalized time",
    y=["Gap"],
    style=['-'],
    color=['b'], markevery=20, lw=5)
apply_fig_settings(gap_time)
draw_open_close(gap_time)
gap_time.set_xlabel("t/T", fontsize=axis_label_size)
gap_time.set_ylabel("Gap width (mm)", fontsize=axis_label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/gap_time_domaion.png", format='png')
plt.show()

plt.plot(freq, np.abs(power), 'b', freq_gap, np.abs(
    power_gap), 'r', lw=4)
plt.plot(freq, resonance, 'k--', lw=4)
plt.yscale("log")
plt.xlim([0, 2000])
plt.ylim([1e-12, 1e3])
plt.xlabel("Frequency (Hz)", fontsize=28)
plt.ylabel("PSD", fontsize=28)
plt.grid()
plt.legend(["$p_{m}$ ($Pa^2/Hz$)", "$h_{g}$ ($mm^2/Hz$)"], fontsize=28)
plt.tight_layout()
plt.savefig(meta_data.output_dir +
            "/pressure_frequency_domain.png", format='png')
plt.show()

# All time gap history
gap_time_full = gap_data.plot(
    x="Time",
    y=["Gap"],
    style=['-'],
    color=['b'], lw=3, figsize=(width*1.5, height*0.36))
plt.locator_params(axis='y', nbins=8)
gap_time_full.tick_params(direction='in', length=20,
                          width=2, top=True, right=True)
gap_time_full.get_legend().remove()
gap_time_full.grid()
gap_time_full.set_xlim([0.0, 0.24])
gap_time_full.set_ylim([0.0, 1.0])
gap_time_full.set_xlabel("Time (s)", fontsize=axis_label_size)
gap_time_full.set_ylabel("h (mm)", fontsize=axis_label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir + "/gap_time_full.png", format='png')
plt.show()
