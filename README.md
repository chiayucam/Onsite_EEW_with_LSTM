# Onsite_EEW_with_LSTM
A visualization tool for testing EEW LSTM model

# Required packages
* scipy 1.4.1
* tensorflow 2.1.0

# Input format and input shape
>The program reads numpy array files (.npy) as input, the shape of the array should be specified as (*, 3)
where * is a variable amount of timesteps and 3 stands for the 3 component (Z, N, E) strong motion record (acceleration in the unit of Gal).

# Output
>The lower figure graphs the probability of the following seismic activity exceeding 80 Gal accordingly to the input
