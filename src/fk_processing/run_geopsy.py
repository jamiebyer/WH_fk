import subprocess
import obspy
import subprocess
from multiprocessing import Pool
import datetime
from obspy import read

import pandas as pd
import hvsrpy
import os
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata

from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt


def read_data():
    data_path = "./data/WH01/"
    for f in os.listdir(data_path):
        if not f.endswith(".mseed"):
            continue
        # data = obspy.read(data_path + "0240_WH01.mseed", "MSEED")
        data = obspy.read(data_path + f, "MSEED")
        print(data.traces[0])
    # data.plot()

    df = pd.read_csv(
        data_path + "WH01_loc_corrected_geopsy.txt",
        sep=" ",
        names=["site", "lat", "lon", "empty"],
    )

    plt.scatter(df["lon"], df["lat"])
    plt.show()


def fk_sensitivity_test():
    pass


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


def get_dispersion_curve(max_file):

    # max_file = "./data/WH02/WH02_fine.max" # jeremy's
    # max_file = "./data/WH01/WH01_fine.max" # jeremy's
    # max_file = "./results/WH01/WH01_main.max"
    # max_file = "./results/WH01/WH01_main.max"
    # max_file = "./capon-importedsignals.max"

    # Open the file in read mode
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

    names = [
        "abs_time",
        "frequency",
        # "polarization",
        "slowness",
        "",
        "azimuth",
        "el",
        "no",
        "power",
        "valid",
    ]

    # df = pd.read_csv(max_file, header=ind, sep=" ")
    df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)

    freqs_grid = df["frequency"]
    vels_grid = 1 / df["slowness"]
    # az = df["azimuth"]
    # power = df["power"]

    # compute dispersion curve
    freqs_curve = np.unique(freqs_grid)
    vels_curve = []
    stds_curve = []
    for f in freqs_curve:
        vel = np.median(vels_grid[freqs_grid == f])
        std = np.std(vels_grid[freqs_grid == f])
        vels_curve.append(vel)
        stds_curve.append(std)

    inds = (freqs_curve >= 2) & (freqs_curve <= 8)

    return freqs_curve, np.array(vels_curve)[inds], np.array(stds_curve)[inds]
