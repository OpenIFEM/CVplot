# CVPlot

A visualization tool for [OpenIFEM](https://github.com/OpenIFEM/OpenIFEM) FSI module with control volume analysis. It plots the following integral equations from the `control_volume_analysis.csv` file that `ControlVolumeFSI<dim>` outputs:

 - Mass conservation
 - Momentum equation
 - Mechanical energy balance
 - Generalized Bernoulli equation
 - Gap history
 - VF surface motion animation

How to use:
 1. Install the dependencies using `python3 -m pip install -r requirements.txt` command under the top directory.
 2. Create a directory named `Cases` under the top directory. This is the default location to store the case files and is ignored by git, but feel free to use other locations. 
 3. In your case subfolders, create shortcuts of the python files you need, then copy `control_volume_analysis.csv` file (for control volume analysis) and `solid_trace` directory (for gap size history and VF motion) from [OpenIFEM](https://github.com/OpenIFEM/OpenIFEM) outputs to the same directory.
 4. Specify `plot_settings.yaml` under the same directory. This file controls the plot settings. An example is given in the source files.
 5. In case gap history is needed, run `gap_history_generator.py` first to create `gap.csv`.
 6. Run source files for corresponding analyses.