# Comparison of Rotor Wake Identification and Characterization Methods for the Analysis of Wake Dynamics and Evolution
EW Quon & P Doubrawa

## Analysis workflow
1. `estimate_inflow.ipynb`: get $U(t,z)$ from $x=-2D$, save to `inflow.npz`
2. `track-all_methods.ipynb`: samwich driver
3. `manual_wake_ID.ipynb`: to generate reference data for filtering assessment
4. `clean_up_trajectories.ipynb`: to remove outliers, interpolate, and filter the wake trajectories
