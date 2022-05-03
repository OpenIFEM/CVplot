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
# superior - VT side
gap_superior = np.zeros(total_timesteps + 1)
# inferior - trachea side
gap_inferior = np.zeros(total_timesteps + 1)
gap_time = np.zeros(total_timesteps + 1)
tip_x = np.zeros(total_timesteps + 1)
# Read first file and determine inferior and superior nodes
initial_data = pd.read_table(
    f"{boundary_trace_dir}{file_prefix}000001", sep=" ",
    header=0, names=["ID", "X", "Y", "P"], index_col="ID")
# ID list
superior_nodes = []
inferior_nodes = []
for ID, item in initial_data.iterrows():
    if item[0] <= 13.9116:
        inferior_nodes.append(ID)
    else:
        superior_nodes.append(ID)


def superior_IDs(row):
    if row["ID"] in superior_nodes:
        return True
    else:
        return False


def inferior_IDs(row):
    if row["ID"] in inferior_nodes:
        return True
    else:
        return False


for index in range(1, total_timesteps + 1, 1):
    current_filename = f'{boundary_trace_dir}{file_prefix}{index:06}'
    current_data = pd.read_table(
        current_filename, sep=" ",
        header=0, names=["ID", "X", "Y", "P"])
    gap[index] = 1.397 - max(current_data["Y"])
    gap_time[index] = index * 5e-7
    tip_x[index] = current_data.at[8, "X"]
    # Superior gap
    superior = current_data.apply(superior_IDs, axis=1)
    gap_superior[index] = 1.397 - max(current_data[superior]["Y"])
    # Superior gap
    inferior = current_data.apply(inferior_IDs, axis=1)
    gap_inferior[index] = 1.397 - max(current_data[inferior]["Y"])
    # Print progress
    total_length = 50
    progress = int(np.floor(index / (total_timesteps + 1) * total_length))
    output = f"Reading progress: ({index}/{total_timesteps + 1})["
    output = output + progress * "█" + (total_length - progress) * " " + "]"
    print(output, end='\r')
print()
print("Save data...")

# Save to a file
save_data = pd.DataFrame(data={"Time": gap_time, "Gap": gap, "Tip_X": tip_x,
                               "Superior": gap_superior, "Inferior": gap_inferior})
save_data.to_csv(meta_data.working_dir + "/gap.csv", index=None)
