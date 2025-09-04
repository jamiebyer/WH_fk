import numpy as np
import matplotlib.pyplot as plt

from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import (
    array_transff_wavenumber,
    array_transff_freqslowness,
)

from fk_processing.dispersion_curves import compute_dispersion_curve
from matplotlib.colors import LogNorm

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from fk_processing.dispersion_curves import read_max_file, read_txt_file


# ARRAY RESPONSE


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

    plt.axhline(y=0.5)

    plt.xlabel("k magnitude (rad/m)")
    plt.ylabel("array response")

    plt.suptitle("test array transfer function")
    plt.tight_layout()
    plt.show()


def plot_array_response():
    # https://docs.obspy.org/tutorial/code_snippets/array_response_function.html
    # https://geophydog.cool/post/array_response_function/#__31-the-geometry-effects__

    data_path = "./data/WH01/txt_files/WH01_loc_corrected_geopsy.txt"
    # generate array coordinates
    coords_df = pd.read_csv(
        data_path,
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

    dists = []
    for c1 in coords:
        for c2 in coords:
            if np.sum(np.abs(c1) - np.abs(c2)) == 0.0:
                continue
            dists.append(np.sqrt(np.sum(np.abs(c1 - c2) ** 2)))

    # minumum distance between stations
    d_min = np.min(dists)
    # maximum distance between stations
    d_max = np.max(dists)

    # kmax
    k_max = 2 * np.pi / d_min
    k_min = 2 * np.pi / d_max

    print(d_min, d_max)
    print(k_min, k_max)

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
    # transff = array_transff_freqslowness

    # plot
    plt.subplot(2, 2, 1)
    plt.scatter(coords_df["x"], coords_df["y"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(2, 2, 2)
    plt.pcolor(kx / 1000, ky / 1000, transff.T)

    # plt.xlim(kxmin / 1000, kxmax / 1000)
    # plt.ylim(kymin / 1000, kymax / 1000)

    plt.Circle((0, 0), k_max / 1000, color="r")

    plt.xlabel("k_x (rad/m)")
    plt.ylabel("k_y (rad/m)")
    plt.colorbar(label="array response")

    plt.subplot(2, 2, 3)
    for yind in range(len(ky)):
        k_mag = np.sqrt(kx**2 * ky[yind] ** 2)
        inds = np.argsort(k_mag)
        # plt.scatter(
        #    k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005, s=3
        # )
        plt.plot(k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005)

    plt.axvline(x=k_min / 1000)
    plt.axvline(x=k_max / 1000)

    plt.axhline(y=0.5)

    plt.xlabel("k magnitude")
    plt.ylabel("array response")

    plt.suptitle("WH01 array transfer function")
    plt.tight_layout()
    # plt.show()


# DISPERSION CURVES


def plot_computed_dispersion_curve(max_file, f_min, f_max):
    """
    Plot dispersion curve from max file.
    """
    freqs_grid, vels_grid, freqs, vels, stds = compute_dispersion_curve(
        max_file, f_min, f_max
    )

    freq_bins = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)), 75)
    vel_bins = np.logspace(np.log10(np.min(vels)), np.log10(np.max(vels)), 100)

    # plot frequency and velocity 2D histogram
    plt.hist2d(
        freqs_grid,
        vels_grid,
        bins=[
            freq_bins,
            vel_bins,
        ],
        cmap="coolwarm",
        norm=LogNorm(),
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.ylim([190, 2100])

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.colorbar(label="counts")

    # plot dispersion curve with errors
    # plt.plot(freqs_curve, vels_curve)

    plt.errorbar(
        freqs,
        vels,
        stds,
        c="black",
        alpha=0.5,
        elinewidth=1,
        barsabove=True,
    )

    plt.grid(True)

    plt.title("computed dispersion curve")
    plt.tight_layout()
    plt.show()


def plot_geopsy_dispersion_curve(max_path, txt_path):
    """
    read dispersion curve
    """

    df_max = read_max_file(max_path)
    df_txt = read_txt_file(txt_path)

    # max file

    freqs = df_max["frequency"]
    vels = 1 / df_max["slowness"]
    # az = df_max["azimuth"]
    # power = df_max["power"]
    #

    freq_bins = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)), 75)
    vel_bins = np.logspace(np.log10(np.min(vels)), np.log10(np.max(vels)), 100)
    # plot frequency and velocity 2D histogram
    plt.hist2d(
        freqs,
        vels,
        bins=[
            freq_bins,
            vel_bins,
        ],
        cmap="coolwarm",
        norm=LogNorm(),
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.ylim([190, 2100])

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.colorbar(label="counts")

    plt.grid(True)

    # txt file

    # plt.scatter(df_txt["frequency"], 1 / df_txt["slowness"], c="black", s=5)
    plt.errorbar(
        df_txt["frequency"],
        1 / df_txt["slowness"],
        1 / (df_txt["percent_error"] * df_txt["slowness"]),
        c="black",
        alpha=0.5,
    )

    plt.title("geopsy dispersion curve")

    plt.show()
