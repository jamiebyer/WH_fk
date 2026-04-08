import subprocess
import obspy
import subprocess
from multiprocessing import Pool

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

from inversion.data import SyntheticData, FieldData


def run_geopsy():
    geopsy_fk_path = "./geopsypack-src-3.5.2/bin/geopsy-fk"
    # max2curve_path = "./geopsypack-src-3.5.2/bin/max2curve"
    # gpviewmax_path = "./geopsypack-src-3.5.2/bin/gpviewmax"

    data_path = "./data/WH01/"
    stations = [
        "0240",
        "0252",
        "0253",
        "0424",
        "0526",
        "TP01",
        "TP02",
        "TP03",
        "TP04",
        "TP05",
        "TP06",
        "TP07",
        "TP08",
        "TP09",
        "TP10",
    ]
    file_list = [data_path + s + "_WH01.mseed" for s in stations]

    # print(hvsrpy.read(file_list))
    # print(hvsrpy.read_single(file_list[0]))
    # subprocess.run([geopsy_fk_path, "./data/WH02/geopsy_signal.gpy"], shell=True)

    # print(os.path.exists("./Mirandola.gpy"))
    # subprocess.run([geopsy_fk_path, "-db ./Mirandola.gpy"], shell=True)
    # subprocess.run([geopsy_fk_path, "-db" "./data/WH01/WH01.gpy"])

    # create database, set receivers
    # utm_zone x y station_name
    # subprocess.run([geopsy_fk_path, file_list], shell=True)

    # run beamforming
    # geopsy-fk -db Mirandola.gpy -group C_135_405-Z -param limits.param

    """
    subprocess.run(
        # [gpviewmax_path, "./data/WH02/WH02_fine.max", "-e", "PNG"],
        [gpviewmax_path, "./data/WH02/WH02_fine.max", "-type", "FK"],
        # [gpviewmax_path, "./data/WH02/WH02_fine.max", "-p", "l", "-e "filename"", "-f", "PNG"],
        shell=True,
    )
    """


def read_txt_file(txt_path):
    # df = pd.read_csv(
    #     data_path + "WH01_loc_corrected_geopsy.txt",
    #     sep=" ",
    #    names=["site", "lat", "lon", "empty"],
    # )

    ###

    # in_path = "./data/WH02/WH02_curve_fine.txt"

    names = ["frequency", "slowness", "percent_error", "unknown", "valid"]
    df = pd.read_csv(txt_path, sep="\s+", names=names)
    return df


def read_max_file(max_file):
    """
    Plot geopsy ".max" file.
    Gives time, frequency, slowness, azimuth, power, --
    """
    # read max file
    # determine how many lines to skip when reading pd dataframe
    with open(max_file, "r") as file:
        # Read the first line
        line = file.readline()
        ind = 0
        while line:
            if "# BEGIN DATA" in line:
                ind += 3
                break
            line = file.readline()  # Read the next line
            ind += 1

    # column names (slightly different for old max files)
    names = [
        "abs_time",
        "frequency",
        # "polarization",
        "slowness",
        "azimuth",
        "",
        "el",
        "no",
        "power",
        "valid",
    ]

    # read ".max" file as dataframe
    df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)
    return df


def compute_dispersion_curve(df, err_thresh=None, freq_outliers=[], vel_outliers=[]):
    """
    compute dispersion curve from .max file df.
    get median and std for each frequency.
    minimum threshold for error.
    """
    freqs_grid = df["frequency"]
    vels_grid = 1 / df["slowness"]
    # az = df["azimuth"]
    # power = df["power"]

    # compute dispersion curve
    freqs_curve = np.unique(freqs_grid)
    vel_meds_curve = []
    vel_means_curve = []
    stds_curve = []
    # for each frequency, save the median velocity, and
    # compute the standard deviation
    for f in freqs_curve:
        vels = vels_grid[freqs_grid == f]
        if len(freq_outliers) > 0:
            ind = np.argmin(np.abs(freq_outliers - f))

            if np.abs(freq_outliers[ind] - f) < 0.01:
                vels = vels[vels < vel_outliers[ind]]

        vel_med = np.median(vels)
        vel_mean = np.mean(vels)
        std = np.std(vels)
        vel_meds_curve.append(vel_med)
        vel_means_curve.append(vel_mean)
        stds_curve.append(std)

    # error threshold
    if err_thresh is not None:
        ind = np.argmin(np.abs(freqs_curve - err_thresh))
        stds_curve[ind:] = (len(stds_curve) - ind) * [stds_curve[ind]]

    return (
        freqs_grid,
        vels_grid,
        freqs_curve,
        np.array(vel_meds_curve),
        np.array(vel_means_curve),
        np.array(stds_curve),
    )


def peak_picking():
    # getting the mode within a range...
    pass


def setup_data(site):
    p = "./results/curves/curve-"+site+"-1C.csv"
    df = pd.read_csv(p)
    freqs, vels, stds = df["freqs"].values, df["vels"].values, df["stds"].values

    periods = np.flip(1 / freqs)
    phase_vels = np.flip(vels / 1000)
    stds = np.flip(stds / 1000)
    data = FieldData(periods, phase_vels, stds)

    return data, freqs, vels
