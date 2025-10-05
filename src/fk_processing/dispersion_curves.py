import subprocess
import obspy
import subprocess
from multiprocessing import Pool

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt


def run_geopsy():
    geopsy_fk_path = "./geopsypack-src-3.5.2/bin/geopsy-fk"
    gpviewmax_path = "./geopsypack-src-3.5.2/bin/max2curve"
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
    # subprocess.run([geopsy_fk_path, "-db" "./data/Mirandola.gpy"])

    # create database, set receivers
    # utm_zone x y station_name
    # subprocess.run([geopsy_fk_path, file_list], shell=True)

    # list parameters
    # geopsy-fk -param-example
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
        "polarization",
        "slowness",
        # "",
        "azimuth",
        "el",
        "no",
        "power",
        "valid",
    ]

    # read ".max" file as dataframe
    df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)
    return df


def compute_dispersion_curve(df, f_min, f_max):
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
        vel_med = np.median(vels_grid[freqs_grid == f])
        vel_mean = np.mean(vels_grid[freqs_grid == f])
        std = np.std(vels_grid[freqs_grid == f])
        vel_meds_curve.append(vel_med)
        vel_means_curve.append(vel_mean)
        stds_curve.append(std)

    # save the frequencies between frequency bounds
    inds = (freqs_curve >= f_min) & (freqs_curve <= f_max)

    return (
        freqs_grid,
        vels_grid,
        freqs_curve,
        np.array(vel_meds_curve),
        np.array(vel_means_curve),
        np.array(stds_curve),
        inds,
    )
