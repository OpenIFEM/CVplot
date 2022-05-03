import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from meta_data import CVMetaData

from pressure import create_pressure_frame
from mass import create_mass_frame
from energy import create_energy_frame
from momentum import create_momentum_frame

from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

# Global variables
S = 1.397  # Cross section area (height in 2D)
c = 34000  # Sound speed
rho = 1.3e-3  # Air density
to_newton = 1e-5  # Convert from dyne to Newton
to_pa = 1e-1  # Convert from dyne/cm^2 to Pascal
to_watt = 1e-7  # Convert from erg/s to Watt
to_mm = 1e1  # Convert from cm to mm

meta_data = CVMetaData(sys.argv)
documents = meta_data.documents

data_collection = {}
# Initialize efficiency key first
data_collection["Efficiency_mean"] = []
gap_data_all = pd.DataFrame()
for lung_pressure, path in meta_data.cases.items():
    cv_data = pd.read_csv(path + "control_volume_analysis.csv", header=0)
    # Time
    time = cv_data["Time"]
    # Time span
    normalized_timespan = [0.0, meta_data.n_period]
    time_to_plot = meta_data.timespan[1] - meta_data.timespan[0]
    T_cycle = time_to_plot/meta_data.n_period
    cv_data["Normalized time"] = (
        cv_data["Time"] - meta_data.timespan[0])/T_cycle

    def filter_period(data_to_filter: pd.DataFrame):
        filtered = data_to_filter[(data_to_filter["Normalized time"] > 0)
                                  & (data_to_filter["Normalized time"] < meta_data.n_period)]
        return filtered

    def statistics(array):
        return (np.mean(array.to_numpy()), np.std(array.to_numpy()))

    cv_pressure = filter_period(create_pressure_frame(cv_data))
    cv_mass = filter_period(create_mass_frame(cv_data))
    cv_energy = filter_period(create_energy_frame(cv_data, documents))
    cv_momentum = filter_period(create_momentum_frame(cv_data))
    cv_merged = cv_pressure.merge(cv_mass, how="inner",
                                  on="Time", suffixes=[None, "_copy"]
                                  ).merge(cv_energy,
                                          how="inner",
                                          on="Time",
                                          suffixes=[None, "_copy"]
                                          ).merge(cv_momentum, how="inner", on="Time", suffixes=[None, "_copy"])
    # Additional terms
    cv_merged["Inlet pressure work"] = cv_data["Inlet pressure work"] * to_watt
    cv_merged["Outlet pressure work"] = cv_data["Outlet pressure work"] * to_watt
    cv_merged["-2P_D^-Q_D"] = -cv_merged["2P_D^-Q_D"]
    cv_merged["Q_A+Q_D"] = cv_merged["Radiated pressure"] / to_pa * S / rho / c
    for label, content in cv_merged.items():
        # Skip time and duplicate
        if label == "Time" or label == "Normalized time" or label[-5:] == "_copy":
            continue
        # Compute mean and stdev
        mean, stdev = statistics(cv_merged[label])
        # Append data
        mean_name = label + "_mean"
        stdev_name = label + "_stdev"
        for name in [mean_name, stdev_name]:
            if name not in data_collection.keys():
                data_collection[name] = []
        data_collection[mean_name].append(mean)
        data_collection[stdev_name].append(stdev)

    # Gap data
    gap_data = pd.read_csv(path + "gap.csv", header=0)
    gap_data_all["Time"] = gap_data["Time"].copy()
    gap_data_all[f"{lung_pressure}"] = gap_data["Gap"].copy() * to_mm
    # Efficiency
    data_collection["Efficiency_mean"].append(-data_collection["Acoustic output_mean"][-1] /
                                              data_collection["Pressure input_mean"][-1])

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
plt.rcParams["font.size"] = label_size


def plot_stats(y_axis_names: list, x_axis_name: str, linespecs: list,
               colors: list, errorbar: bool, x_stdev: bool = False,
               y_stdev: bool = False):
    x_axis_name = x_axis_name + "_mean" if x_stdev == False else x_axis_name + "_stdev"
    for i in range(len(y_axis_names)):
        name = y_axis_names[i]
        line = linespecs[i]
        color = colors[i]
        mean_name = name + "_mean"
        stdev_name = name + "_stdev"
        name_used = mean_name if y_stdev == False else stdev_name
        if errorbar == True:
            plt.errorbar(
                data_collection[x_axis_name], data_collection[mean_name],
                yerr=data_collection[stdev_name],
                capsize=20, capthick=5, label=name,
                linestyle=line, marker=">", color=color,
                markersize=35, lw=3)
        else:
            plt.plot(
                data_collection[x_axis_name], data_collection[name_used],
                linestyle=line, marker=">", color=color,
                markersize=35, lw=5)
    # get handles
    handles, labels = plt.gca().get_legend_handles_labels()
    # remove the errorbars
    if errorbar:
        handles = [h[0] for h in handles]
    return handles


def apply_fig_settings(fig):
    plt.locator_params(axis='y', nbins=8)
    fig.tick_params(direction='in', length=20,
                    width=2, top=True, right=True)


height_scaling = 1.01
plt.rcParams["figure.subplot.bottom"] = (
    plt.rcParams["figure.subplot.bottom"] + height_scaling - 1) / height_scaling
# Plot
# Fig: mean subglottal and transglottal pressure vs mean volume flow
fig = plt.figure()
pa_tgp_vs_volume_flow = fig.add_subplot(1, 1, 1)
y_axis_names = ["P_A", "TGP"]
x_axis_name = "Q_A+Q_D"
handles = plot_stats(y_axis_names, x_axis_name, ["-", "--"], ["b", "g"], True)
apply_fig_settings(pa_tgp_vs_volume_flow)
pa_tgp_vs_volume_flow.legend(
    handles,
    [r"$\overline{p_\mathrm{A}}$",
     r"$\overline{p_\mathrm{A}} - \overline{p_\mathrm{D}}$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, numpoints=1, frameon=False)
plt.xlabel(
    r"$\overline{Q_\mathrm{A}} + \overline{Q_\mathrm{D}} (cm^3/s)$")
plt.ylabel("Pressure (Pa)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_pa_tgp_vs_volume_flow.png", format='png')
plt.show()

# Fig: mean subglottal built-up pressure and mean subglottal
# pressure vs mean subglottal pressure
fig = plt.figure()
pap_vs_pa = fig.add_subplot(1, 1, 1)
y_axis_names = ["Entrance built-up pressure", "P_A"]
x_axis_name = "P_A"
handles = plot_stats(y_axis_names, x_axis_name, ["-", "--"], ["b", "g"], True)
apply_fig_settings(pap_vs_pa)
pap_vs_pa.legend(
    handles,
    [r"$2\overline{p_\mathrm{A}^+}$",
     r"$\overline{p_\mathrm{A}}$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(r"$\overline{p_\mathrm{A}}$ (Pa)")
plt.ylabel("Pressure (Pa)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_pap_vs_sub_p.png", format='png')
plt.show()

# Fig: Efficiency vs. driving pressure
fig = plt.figure()
efficiency_vs_pa = fig.add_subplot(1, 1, 1)
y_axis_names = ["Efficiency"]
x_axis_name = "P_A^+"
handles = plot_stats(y_axis_names, x_axis_name, ["-"], ["b"], False)
apply_fig_settings(efficiency_vs_pa)
ylabels = [format(label, '.1%') for label in efficiency_vs_pa.get_yticks()]
efficiency_vs_pa.set_yticklabels(ylabels)
plt.xlabel(
    r"$\overline{p_\mathrm{L}}$ (Pa)")
plt.ylabel("Laryngeal Efficiency")
plt.savefig(meta_data.output_dir +
            "/cv_stats_efficiency.png", format='png')
plt.show()


# Change size to fit legend
width_scaling = 1.25
plt.rcParams["figure.figsize"] = [width*width_scaling, height*height_scaling]
plt.rcParams["figure.subplot.left"] = (
    meta_data.size["left"] + (width_scaling - 1) / 2) / width_scaling
plt.rcParams["figure.subplot.right"] = (
    meta_data.size["right"] + (width_scaling - 1) / 2) / width_scaling


# Fig: TGP force, drag force, radiated pressure force
fig = plt.figure()
tgp_drag_volume_source_mean = fig.add_subplot(1, 1, 1)
y_axis_names = ["TGP force", "-VF drag", "Radiated pressure force"]
x_axis_name = "Volume source"
handles = plot_stats(y_axis_names, x_axis_name, [
                     "-", "--", "-."], ["b", "g", "r"],
                     False)
apply_fig_settings(tgp_drag_volume_source_mean)
tgp_drag_volume_source_mean.legend(
    [r"TGP force",
     r"VF drag",
     r"$-\frac{\rho c}{S_\mathrm{VT}}(\overline{Q_\mathrm{A}}+\overline{Q_\mathrm{D}})$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(
    r"$\rho c(\overline{Q_\mathrm{A}} + \overline{Q_\mathrm{D}})$(N)")
plt.ylabel("Momentum Equation Terms (N)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_tgp_drag_volume_source_mean.png", format='png')
plt.show()

fig = plt.figure()
tgp_drag_volume_source_stdev = fig.add_subplot(1, 1, 1)
y_axis_names = ["TGP force", "-VF drag", "Radiated pressure force"]
x_axis_name = "Volume source"
handles = plot_stats(y_axis_names, x_axis_name, [
                     "-", "--", "-."], ["b", "g", "r"],
                     False, x_stdev=True, y_stdev=True)
apply_fig_settings(tgp_drag_volume_source_stdev)
tgp_drag_volume_source_stdev.legend(
    [r"$\sigma$: TGP force",
     r"$\sigma$: VF drag",
     r"$\sigma$: $-\frac{\rho c}{S_\mathrm{VT}}(Q_\mathrm{A}+Q_\mathrm{D})$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(
    r"$\sigma$: $\rho c(Q_\mathrm{A} + Q_\mathrm{D})$(N)")
plt.ylabel("Momentum Equation Terms (N)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_tgp_drag_volume_source_stdev.png", format='png')
plt.show()


def update_ylabels_and_lim(fig):
    if ("ylim" in item for item in documents["power stats"]):
        ylim = next(d for i, d in enumerate(
            documents["power stats"]) if "ylim" in d)
        fig.set_ylim(ylim["ylim"])
    ylabels = [format(label, '.1f') for label in fig.get_yticks()]
    fig.set_yticklabels(ylabels)


# Fig: Flow work at inlet face vs. driving pressure
fig = plt.figure()
inlet_works_vs_drive_p = fig.add_subplot(1, 1, 1)
y_axis_names = ["Inlet pressure work", "2P_A^+Q_A", "Acoustic loss"]
x_axis_name = "P_A^+"
handles = plot_stats(y_axis_names, x_axis_name, [
                     "-", "--", "-."], ["b", "g", "r"], True)
apply_fig_settings(inlet_works_vs_drive_p)
update_ylabels_and_lim(inlet_works_vs_drive_p)
inlet_works_vs_drive_p.legend(
    handles,
    [r"$\overline{p_\mathrm{A}Q_\mathrm{A}}$",
     r"$2\overline{p_\mathrm{A}^+Q_\mathrm{A}}$",
     r"$-\frac{\rho c}{S_\mathrm{VT}}\overline{Q_\mathrm{A}^2}$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(
    r"$\overline{p_\mathrm{L}}$ (Pa)")
plt.ylabel("Flow Work (Watts)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_inlet_work_decomposition.png", format='png')
plt.show()

# Fig: Flow work at outlet face vs. driving pressure
fig = plt.figure()
outlet_works_vs_drive_p = fig.add_subplot(1, 1, 1)
y_axis_names = ["Outlet pressure work", "2P_D^-Q_D", "Acoustic output"]
x_axis_name = "P_A^+"
handles = plot_stats(y_axis_names, x_axis_name, [
                     "-", "--", "-."], ["b", "g", "r"], True)
apply_fig_settings(outlet_works_vs_drive_p)
update_ylabels_and_lim(outlet_works_vs_drive_p)
outlet_works_vs_drive_p.legend(
    handles,
    [r"$\overline{p_\mathrm{D}Q_\mathrm{D}}$",
     r"$-2\overline{p_\mathrm{D}^-Q_\mathrm{D}}$",
     r"$-\frac{\rho c}{S_\mathrm{VT}}\overline{Q_\mathrm{D}^2}$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(
    r"$\overline{p_\mathrm{L}}$ (Pa)")
plt.ylabel("Flow Work (Watts)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_outlet_work_decomposition.png", format='png')
plt.show()

# Fig: Flow work vs. driving pressure
fig = plt.figure()
tot_works_vs_drive_p = fig.add_subplot(1, 1, 1)
y_axis_names = ["Driving pressure work",
                "Pressure input", "Net acoustic power flow"]
x_axis_name = "P_A^+"
handles = plot_stats(y_axis_names, x_axis_name, [
                     "-", "--", "-."], ["b", "g", "r"], True)
apply_fig_settings(tot_works_vs_drive_p)
update_ylabels_and_lim(tot_works_vs_drive_p)
tot_works_vs_drive_p.legend(
    handles,
    [r"$\overline{p_\mathrm{A}Q_\mathrm{A}} - \overline{p_\mathrm{D}Q_\mathrm{D}}$",
     r"$2(\overline{p_\mathrm{A}^+Q_\mathrm{A}} - \overline{p_\mathrm{D}^-Q_\mathrm{D}})$",
     r"$-\frac{\rho c}{S_\mathrm{VT}}(\overline{Q_\mathrm{A}^2} + \overline{Q_\mathrm{D}^2})$"],
    bbox_to_anchor=(1.0, 0.5),
    loc='center left', ncol=1, labelspacing=2, frameon=False)
plt.xlabel(
    r"$\overline{p_\mathrm{L}}$ (Pa)")
plt.ylabel("Flow Work (Watts)")
plt.savefig(meta_data.output_dir +
            "/cv_stats_total_work_decomposition.png", format='png')
plt.show()

# Get a list of cases for legend
case_inputs = [r"$p_\mathrm{L}$" + f"= {i} Pa" for i in meta_data.cases.keys()]
# All time gap history
gap_time_all_zoomed_out = gap_data_all.plot(
    x="Time", lw=2, figsize=(width*1.2, height*0.5))
plt.locator_params(axis='y', nbins=8)
gap_time_all_zoomed_out.tick_params(direction='in', length=20,
                                    width=2, top=True, right=True)
gap_time_all_zoomed_out.legend(case_inputs,
                               bbox_to_anchor=(1.0, 0.5),
                               loc='center left', ncol=1, frameon=False)
gap_time_all_zoomed_out.set_xlim([0.0, 0.15])
gap_time_all_zoomed_out.set_ylim([-0.08, 1.3])
gap_time_all_zoomed_out.set_xlabel("Time (s)", fontsize=label_size)
gap_time_all_zoomed_out.set_ylabel("h (mm)", fontsize=label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir +
            "/gap_time_all_zoomed_out.png", format='png')
plt.show()

# All time gap history
gap_time_all_zoomed_in = gap_data_all.plot(
    x="Time", lw=3, figsize=(width, height))
plt.locator_params(axis='y', nbins=8)
gap_time_all_zoomed_in.tick_params(direction='in', length=20,
                                   width=2, top=True, right=True)
gap_time_all_zoomed_in.legend(case_inputs,
                              bbox_to_anchor=(1.0, 0.5),
                              loc='center left', ncol=1,
                              labelspacing=2, frameon=False)
# gap_time_all_zoomed_in.set_xlim(meta_data.timespan)
gap_time_all_zoomed_in.set_xlim(0.135, 0.15)
gap_time_all_zoomed_in.set_ylim([-0.08, 1.3])
gap_time_all_zoomed_in.set_xlabel("Time (s)", fontsize=label_size)
gap_time_all_zoomed_in.set_ylabel("h (mm)", fontsize=label_size)
# Save the plot
plt.tight_layout()
plt.savefig(meta_data.output_dir +
            "/gap_time_all_zoomed_in.png", format='png')
plt.show()

# Compute the limit for laryngeal efficiency
slope_input = np.polyfit(
    data_collection["Driving pressure work_mean"], data_collection["Pressure input_mean"], 1)
slope_output = np.polyfit(
    data_collection["Driving pressure work_mean"], data_collection["Acoustic output_mean"], 1)
print(f"Efficiency limit is {-slope_output[0]/slope_input[0]:2%}")
