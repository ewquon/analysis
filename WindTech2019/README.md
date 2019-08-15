# Comparison of Rotor Wake Identification and Characterization Methods for the Analysis of Wake Dynamics and Evolution
EW Quon & P Doubrawa

## Analysis workflow
1. `estimate_inflow.ipynb`: get $U(t,z)$ from $x=-2D$, save to `inflow.npz`
2. `track-all_methods.ipynb`: samwich driver
3. `manual_wake_ID.ipynb`: to generate reference data for filtering assessment
4. `clean_up_trajectories.ipynb`: to remove outliers, interpolate, and filter the wake trajectories
5. `calculate_wake_MFoR.ipynb`: calculate `mean_wake_mfor.nc` and `wake_stdev_mfor.nc`
6. `analyze_wake_MFoR`: check size and max deficit in meandering frame of reference
