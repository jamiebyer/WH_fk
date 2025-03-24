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


def plot_max_file(max_file):
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
        "azimuth",
        "",
        "ellipticity",
        "noise",
        "power",
        "valid",
    ]

    data_types = {
        "abs_time": float,
        "frequency": float,
        "polarization": str,
        "slowness": float,
        "azimuth": float,
        "ellipticity": float,
        "noise": float,
        "power": float,
        "valid": int,
    }

    # df = pd.read_csv(max_file, header=ind, sep=" ")
    df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)

    """
    freqs_array = np.linspace(np.min(freqs), np.max(freqs), 1000)
    vels_array = np.linspace(np.min(vels), np.max(vels), 1000)

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1000))

    # Fit and transform the data
    freqs = scaler.fit_transform(freqs.values.reshape(-1, 1))
    vels = scaler.fit_transform(vels.values.reshape(-1, 1))

    # grid_x, grid_y = np.mgrid[freqs_array, vels_array]
    grid_x, grid_y = np.mgrid[0:1:1000, 0:1:1000]
    points = np.array([freqs.squeeze(), vels.squeeze()])
    # values = np.ones(len(freqs))
    values = df["power"]

    grid_z0 = griddata(points.T, values, (grid_x, grid_y), method="nearest")
    # grid_z1 = griddata(points, values, (grid_x, grid_y), method="linear")
    # grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    plt.imshow(grid_z0.T, origin="lower")


    """

    freqs = df["frequency"]
    vels = 1 / df["slowness"]
    az = df["azimuth"]
    power = df["power"]

    # """
    k_min, k_max = 0.087597, 0.0897466  # rad/m
    vel_min = k_min / (2 * np.pi * freqs)
    vel_max = k_max / (2 * np.pi * freqs)
    plt.plot(freqs, 1 / vel_min)
    plt.plot(freqs, 1 / vel_max)
    plt.xscale("log")
    plt.show()
    # """

    # print(np.min(vels), np.max(vels))
    plt.subplot(3, 1, 1)
    plt.hist2d(freqs, vels, bins=200, cmap="coolwarm", norm=LogNorm())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("freqs")
    plt.ylabel("velocity")

    k_min, k_max = 0.087597, 0.0897466  # rad/m

    vel_min = k_min / (2 * np.pi * freqs)
    vel_max = k_max / (2 * np.pi * freqs)
    plt.plot(freqs, vel_min)
    plt.plot(freqs, vel_max)

    plt.colorbar()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.hist2d(freqs, az, bins=200, cmap="coolwarm", norm=LogNorm())
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("freqs")
    plt.ylabel("azimuth")
    plt.colorbar()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.hist2d(freqs, power, bins=200, cmap="coolwarm", norm=LogNorm())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("freqs")
    plt.ylabel("power")

    plt.colorbar()
    plt.grid()

    plt.tight_layout()

    # abs_time frequency polarization slowness azimuth ellipticity noise power valid
    # plt.plot(df["abs_time"], df["slowness"])
    # plt.subplot(2, 1, 1)

    # plt.scatter(df["frequency"], df["slowness"], c=df["power"], s=5)
    # plt.xscale("log")
    # plt.xlabel("frequency")
    # plt.colorbar()

    # plt.subplot(2, 1, 2)
    # plt.scatter(df["frequency"], df["ellipticity"])
    # plt.xscale("log")

    plt.show()


def plot_dispersion_curve():
    in_path = "./data/WH01/WH01_curve_fine.txt"

    names = ["frequency", "slowness", "unknown_1", "unknown_2", "valid"]
    df = pd.read_csv(in_path, sep="\s+", names=names)

    plt.scatter(df["frequency"], 1 / df["slowness"])

    plt.xscale("log")
    plt.yscale("log")

    plt.show()


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

    # max_file = "./data/WH02/WH02_fine.max"
    max_file = "./results/capon-importedsignals.max"
    plot_max_file(max_file)

    # for 3-component
    # PROCESS_TYPE=RTBF
