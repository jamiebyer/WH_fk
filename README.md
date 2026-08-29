# Whitehorse ambient seismic noise arrays

## Description
F-k beamforming of recordings of ambient seismic noise from arrays at 4 different sites in Whitehorse, Yukon to extract Rayleigh wave dispersion curves.

Link to other related repos. (mapping, mcmc)

## Data
- 4 WH stations (add information about instruments used at each site, dates of recordings, lengths of recordings, etc.)
- Miniseed files of array recordings under `./data/WH0*`
- 1-C and 3-C recordings for WH01 and WH02, 3-C recordings for WH03 and WH04.

## Processing
### Raw data
- The raw data are 3-C recordings in miniseed files. `./data/WH0*`
- .txt files with coordinates

- Read files and perform processing with obspy.
- For sites WH03 and WH04, slice out the middle of the recording, where there is not noise from the setup/takedown of instruments. Save the sliced data to a new file.


### HVSR
- HVSR using hvsrpy

### F-k beamforming

- F-k beamforming performed using geopsy.
- Param files used to run beamforming at `./params/`
- F-k beamforming results are saved in a .max file.

1. create database from miniseeds and coordinates ".gpy"
2. run fk-beamforming
`./geopsypack-src-3.5.2/bin/geopsy-fk`
`geopsy-fk -db Mirandola.gpy -group C_135_405-Z -param limits.param`


### Dispersion curve picking
- Select points on the 2d histogram of f-k beamforming results to form a polygon.
- The data within the polygon is used for picking the dispersion curve. The dispersion curve at each frequency is the average of velocities in the fullest histogram bin within the polygon.
- The polygon is used to reject outliers in the data, artifacts from the f-k beamforming, and unwanted noise.

### Residuals
- The residuals are a subset of the f-k beamforming results with the dispersion curve subtracted so they show the distribution of noise/error. 
- The frequency range is selected between the kmin/2 and kmax/2 approximations.
    - The kmax/2 cutoff is used for sites WH01 and WH02 since it also removes most of a diagonal artifact from the f-k beamforming. For sites WH03 and WH04, kmax/2 is below the data.
    - kmax/2 and kmin/2 are used to select the frequency range, but to remove the artifact, kmax/2 is used to also remove data.
- The velocity range is selected between 100 and 1200 m/s.

#### Subset data

### Error distribution fitting
- `./processing/distribution_fitting.py`
- Fitting an Asymmetric Laplacian distribution to the residuals of the dispersion curve histogram.
- lambda is inversely related to the spread of the distribution.
- kappa is a parameter describing the skewedness of the distribution.kappa < 1 is positively tailed and kappa > 1 is negatively tailed.

#### Fitting
- To fit the data, lambda/kappa is either:
    - independently found at each frequency
    - a single value found by fitting at all frequencies
    - a single value found by fitting at all frequencies, scaled by the spread/ratio scaling parameter
- Fitting is done using a grid search, and minimizing the least squares of the data.
- The f-k beamforming results at a particular frequency is a distribution of velocities. Fitting is done using the non-negative values of the histograms at each frequency. Minimizing the least-squares difference between these points and the corresponding point on the proposed error distribution.
- `error_fitting_by_full_dataset`: get error distribution parameters by using a grid search to get best fit. Save df to file with quantiles (to compare with spread) 


#### Scaling parameters
- The scale parameter / the spread of the data is quantified as the difference between the 95th and 5th quantile of the data at a particular frequency. This is then smoothed using a 3-point rolling average, and a left- / right-sided average for the endpoints.
- Smoothed using bandwidth parameter from f-k beamforming.

- `get_scaling_params` (in `./src/processing/distribution_fitting.py`): Get 5th and 95th quantiles for the data at each frequency, as well as the spread, ratio, smoothed spread, and smoothed ratio. Save to csv at `./results/curve_fitting/scaling_params/WH0*_scaling_params.csv`

#### Model distributions
- Use fitting to get model distributions used for synthetic data inversions.


## Plotting
List of figures and which file is used to create them.

### Raw data
- Raw data traces (`./src/plotting/raw_data_plotting.py`)

- Plot the vertical component of all instruments. Demean, detrend, and apply a high-pass filter, normalize the traces.
- Plot the frequency content at each site/instrument.

### Array
- Array configuration (`./src/plotting/array_plotting.py`)
- Array response function (`./src/plotting/array_plotting.py`)

### Dispersion curve
- `./plotting/dispersion_curve_plotting.py`
- Plot residuals and data spread
