import sys
import numpy as np
import pandas as pd
import os
from meta_data import CVMetaData

meta_data = CVMetaData(sys.argv)

# Read boundary files for VF gap
half_VT_width = 1.397
boundary_trace_dir = meta_data.working_dir + "/solid_trace/"
file_prefix = "BoundaryTrace-"
total_timesteps = len(os.listdir(boundary_trace_dir))
gap = np.zeros(total_timesteps + 1)
gap_time = np.zeros(total_timesteps + 1)
for index in range(1, total_timesteps + 1, 1):
    current_filename = f'{boundary_trace_dir}{file_prefix}{index:06}'
    current_data = pd.read_table(
        current_filename, sep=" ",
        header=0, names=["ID", "X", "Y", "P"], index_col="ID")
    gap[index] = 1.397 - max(current_data["Y"])
    gap_time[index] = index * 5e-7
    # Print progress
    total_length = 50
    progress = int(np.floor(index / (total_timesteps + 1) * total_length))
    output = f"Reading progress: ({index}/{total_timesteps + 1})["
    output = output + progress * "█" + (total_length - progress) * " " + "]"
    print(output, end='\r')
print()
print("Save data...")

# Save to a file
save_data = pd.DataFrame(data={"Time": gap_time, "Gap": gap})
save_data.to_csv(meta_data.working_dir + "/gap.csv", index=None)
