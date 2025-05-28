import numpy as np
import matplotlib.pyplot as plt

from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import (
    array_transff_wavenumber,
    array_transff_freqslowness,
)

from matplotlib.colors import LogNorm

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


def plot_example_array_response():

    coords = [[-5, 0, 0], [0, 5, 0], [0, 0, 0], [0, -5, 0], [5, 0, 0]]

    coords = np.array(coords) / 1000
    # coordinates in km
    # coords /= 1000.0

    # set limits for wavenumber differences to analyze
    klim = 2500.0
    kxmin = -klim
    kxmax = klim
    kymin = -klim
    kymax = klim
    kstep = klim / 100

    kx = np.arange(kxmin, kxmax + kstep, kstep)
    ky = np.arange(kymin, kymax + kstep, kstep)

    # compute transfer function as a function of wavenumber difference
    transff = array_transff_wavenumber(coords, klim, kstep, coordsys="xy")
    """
    transff = array_transff_freqslowness(
        coords,
        slim=2.0,
        sstep=2 / 100,
        fmin=0.1,
        fmax=40,
        fstep=40 / 100,
        coordsys="xy",
    )
    """

    # plot
    plt.subplot(2, 2, 1)
    plt.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    plt.subplot(2, 2, 2)
    plt.pcolor(kx / 1000, ky / 1000, transff.T)

    plt.xlim(kxmin / 1000, kxmax / 1000)
    plt.ylim(kymin / 1000, kymax / 1000)

    plt.xlabel("k_x (rad/m)")
    plt.ylabel("k_y (rad/m)")
    plt.colorbar(label="array response")

    plt.subplot(2, 2, 3)
    for yind in range(len(ky)):
        k_mag = np.sqrt(kx**2 * ky[yind] ** 2)
        inds = np.argsort(k_mag)
        plt.scatter(
            k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005, s=3
        )

    plt.xlabel("k magnitude (rad/m)")
    plt.ylabel("array response")

    plt.suptitle("test array transfer function")
    plt.tight_layout()
    plt.show()


def plot_array_response():
    # https://docs.obspy.org/tutorial/code_snippets/array_response_function.html
    # https://geophydog.cool/post/array_response_function/#__31-the-geometry-effects__

    # generate array coordinates
    coords_df = pd.read_csv(
        "./data/WH01/txt_files/WH01_loc_corrected_geopsy.txt",
        # "./data/WH02/WH02_loc_corrected_geopsy.txt",
        index_col=0,
        names=["x", "y"],
        sep="\s+",
    )

    coords = np.zeros((len(coords_df), 3))
    coords[:, 0] = coords_df["x"].values
    coords[:, 1] = coords_df["y"].values

    # coordinates in km
    coords /= 1000.0

    # set limits for wavenumber differences to analyze
    klim = 500.0
    kxmin = -klim
    kxmax = klim
    kymin = -klim
    kymax = klim
    kstep = klim / 250.0

    kx = np.arange(kxmin, kxmax + kstep, kstep)
    ky = np.arange(kymin, kymax + kstep, kstep)

    # compute transfer function as a function of wavenumber difference
    transff = array_transff_wavenumber(coords, klim, kstep, coordsys="xy")
    """
    transff = array_transff_freqslowness(
        coords,
        slim=2.0,
        sstep=2 / 100,
        fmin=0.1,
        fmax=40,
        fstep=40 / 100,
        coordsys="xy",
    )
    """

    # plot
    plt.subplot(2, 2, 1)
    plt.scatter(coords_df["x"], coords_df["y"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(2, 2, 2)
    plt.pcolor(kx / 1000, ky / 1000, transff.T)

    plt.xlim(kxmin / 1000, kxmax / 1000)
    plt.ylim(kymin / 1000, kymax / 1000)

    plt.xlabel("k_x (rad/m)")
    plt.ylabel("k_y (rad/m)")
    plt.colorbar(label="array response")

    plt.subplot(2, 2, 3)
    for yind in range(len(ky)):
        k_mag = np.sqrt(kx**2 * ky[yind] ** 2)
        inds = np.argsort(k_mag)
        plt.scatter(
            k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005, s=3
        )

    plt.xlabel("k magnitude")
    plt.ylabel("array response")

    plt.suptitle("WH01 array transfer function")
    plt.tight_layout()
    plt.show()


def plot_txt_file():
    in_path = "./data/WH01/WH01_curve_fine.txt"
    # in_path = "./data/WH02/WH02_curve_fine.txt"

    names = ["frequency", "slowness", "unknown_1", "unknown_2", "valid"]
    df = pd.read_csv(in_path, sep="\s+", names=names)

    plt.subplot(2, 1, 1)
    plt.scatter(df["frequency"], 1 / df["slowness"])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (km/s)")
    plt.show()


def plot_dispersion_curve():

    # max_file = "./data/WH02/WH02_fine.max" # jeremy's
    # max_file = "./data/WH01/WH01_fine.max" # jeremy's
    # max_file = "./results/WH01/WH01_main.max"
    max_file = "./results/WH01/WH01_main.max"
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

    # M = len(np.unique(freqs_grid))
    # N = len(np.unique(vels_grid))

    h, xedges, yedges, _ = plt.hist2d(
        freqs_grid,
        vels_grid,
        bins=[
            np.logspace(np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), 75),
            np.logspace(np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), 100),
        ],
        cmap="coolwarm",
        norm=LogNorm(),
        # rwidth=0.5,  # cmin=1
    )
    # plt.pcolormesh(
    #    power.values.reshape(M, N),
    #    freqs_grid.values.reshape(M, N),
    #    vels_grid.values.reshape(M, N),
    # )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.colorbar()

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
    # plot dispersion curve
    # plt.plot(freqs_curve, vels_curve)
    plt.errorbar(
        freqs_curve[inds],
        np.array(vels_curve)[inds],
        np.array(stds_curve)[inds],
        c="black",
        elinewidth=1,
        barsabove=True,
    )

    plt.grid(True)

    plt.title("WH02 dispersion curve")
    plt.tight_layout()
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

    # df = pd.read_csv(max_file, header=ind, sep=" ")
    df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)
    print(df)

    freqs = df["frequency"]
    vels = 1 / df["slowness"]
    az = df["azimuth"]
    power = df["power"]

    """
    k_min, k_max = 0.087597, 0.0897466  # rad/m
    vel_min = k_min / (2 * np.pi * freqs)
    vel_max = k_max / (2 * np.pi * freqs)
    plt.plot(freqs, 1 / vel_min)
    plt.plot(freqs, 1 / vel_max)
    plt.xscale("log")
    plt.show()
    """

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
