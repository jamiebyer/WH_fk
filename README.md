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
- Polygon picking
- Select points on the 2d histogram of f-k beamforming results to form a polygon.
- The data within the polygon is used for fitting error distributions.
- The polygon is used to reject outliers in the data, artifacts from the f-k beamforming, and unwanted noise.

### Error distribution fitting
- `./processing/distribution_fitting.py`
- Fitting an Asymmetric Laplacian distribution to the residuals of the dispersion curve histogram...
- Fitting with a single value for kappa, and a single value for lambda which is scaled by the spread of the data. Lambda is divided by the scale parameter, since lambda is inversely related to the spread of the distribution.
- The scale parameter / the spread of the data is quantified as the difference between the 95th and 5th quantile of the data at a particular frequency. This is then smoothed using a 3-point rolling average, and a left- / right-sided average for the endpoints.
- Fitting is done using a grid search, and minimizing the least squares of the data.
- The f-k beamforming results at a particular frequency is a distribution of velocities. Fitting is done using the non-negative values of the histograms at each frequency. Minimizing the least-squares difference between these points and the corresponding point on the proposed error distribution.

- Fit an exponential to the spread of the data to use for more parameterization of the data / to create synthetic data.

## Plotting
List of figures and which file is used to create them.

### Raw data
- Raw data traces (`./src/plotting/raw_data_plotting.py`)

- Plot the vertical component of all instruments. Demean, detrend, and apply a high-pass filter, normalize the traces.
- Plot the frequency content at each site/instrument.

### Array
- Array configuration (`./src/plotting/array_plotting.py`)
- Array response function (`./src/plotting/array_plotting.py`)
