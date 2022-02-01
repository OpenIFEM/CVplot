import pandas as pd
from meta_data import CVMetaData
import numpy as np


def smooth(input, input_name, output_name, sample_range):
    input_sum = input[input_name].to_numpy(copy=True)
    for i in range(len(input_sum)):
        if i != 0:
            input_sum[i] = input_sum[i-1]+input_sum[i]*5e-7

    filtered_input_sum = np.zeros(len(input_sum))
    r = sample_range
    for i in range(len(input_sum)):
        filtered_input_sum[i] = input_sum[i]
    for i in range(len(input_sum)):
        if i > r-1 and i < len(input_sum) - (r+1):
            filtered_input_sum[i] = 0.0
            for j in range(r):
                filtered_input_sum[i] += input_sum[i-j] + input_sum[i+j]
            filtered_input_sum[i] = filtered_input_sum[i] / (2*r)

    output = np.zeros(len(input_sum))
    for i in range(len(input_sum)):
        if i > r+3 and i < len(filtered_input_sum) - (r+5):
            output[i] = (0.8 * filtered_input_sum[i+1] - 0.2 * filtered_input_sum[i+2] + 4/105 * filtered_input_sum[i+3]
                         - 1/280 * filtered_input_sum[i+4] - 0.8 * filtered_input_sum[i-1] +
                         0.2 * filtered_input_sum[i-2] -
                         4/105 * filtered_input_sum[i-3]
                         + 1/280 * filtered_input_sum[i-4])/5e-7
    input[output_name] = output


def direct_smooth(input, input_name, output_name, sample_range, smooth_range):
    input_array = input[input_name].to_numpy(copy=True)

    r = sample_range
    output = np.zeros(len(input_array))
    for i in range(smooth_range[0], smooth_range[1], 1):
        if i > r-1 and i < len(input_array) - (r+1):
            output[i] = 0.0
            for j in range(r):
                output[i] += input_array[i-j] + input_array[i+j]
            output[i] = output[i] / (2*r)

    input[output_name] = output


def smooth_data(data: pd.DataFrame, meta_data: CVMetaData, smooth_range: int):
    for index in range(len(data["Normalized time"])):
        if data["Normalized time"][index] < 0:
            smooth_start_index = index
        if data["Normalized time"][index] < meta_data.n_period:
            smooth_end_index = index
    smooth_end_index += smooth_range
    smooth_start_index -= smooth_range

    print(f"Smooth data in [{smooth_start_index} {smooth_end_index}] range...")
    for label, content in data.items():
        if label != "Time" and label != "Normalized time":
            direct_smooth(data, label, label, smooth_range, [
                smooth_start_index, smooth_end_index])
