from smooth import *
from meta_data import CVMetaData
import pandas as pd
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
matplotlib.use("Agg")


# Global variables
S = 1.397  # Cross section area (height in 2D)
c = 34000  # Sound speed
rho = 1.3e-3  # Air density
dt = 5e-7  # Time step size

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

# Read boundary files
boundary_trace_dir = meta_data.working_dir + "/solid_trace/"

# Set up formatting for the movie files
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
# creating a subplot
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')

# Start and end time step
output_interval = 23
file_prefix = "BoundaryTrace-"
start_timestep = int(meta_data.timespan[0] / dt)
end_timestep = int(meta_data.timespan[1] / dt)
n_frames = int((end_timestep - start_timestep + 1) // output_interval) - 1


# Animation function
def animate(i):
    plt.cla()
    timestamp = (i + 1) * output_interval + start_timestep
    current_data = pd.read_table(
        boundary_trace_dir + file_prefix + str(timestamp), sep=" ",
        header=0, names=["ID", "X", "Y", "P"], index_col="ID")
    plt.scatter(current_data["X"], current_data["Y"],
                c=current_data["P"], cmap="rainbow")
    plt.xlim([12.5, 14.5])
    plt.ylim([0.0, 1.5])
    plt.title(f"t = {timestamp * dt:.7f}")


# Initialization function
def init_animation():
    animate(-1)
    plt.colorbar()


# Progress bar for animation save
def progress(current_frame: int, total_frames: int):
    current_frame = current_frame + 1
    total_length = 50
    progress = int(np.floor(current_frame / total_frames * total_length))
    output = f"Saving progress: (frame {current_frame}/{total_frames})["
    output = output + progress * "█" + (total_length - progress) * " " + "]"
    print(output, end='\r')


ani = animation.FuncAnimation(
    fig, func=animate, init_func=init_animation,
    frames=n_frames, interval=40, repeat=False)
ani.save("vf_trace.mp4", writer=writer, progress_callback=progress)
print()
print("Done!")
