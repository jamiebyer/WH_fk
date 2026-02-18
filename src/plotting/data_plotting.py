import numpy as np
import matplotlib.pyplot as plt

from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import (
    array_transff_wavenumber,
    array_transff_freqslowness,
)

from matplotlib.gridspec import GridSpec

import matplotlib.image as img

# import sys

# sys.path.append("../src/")
from processing.dispersion_curves import compute_dispersion_curve, setup_data
from matplotlib.colors import LogNorm

from matplotlib.ticker import ScalarFormatter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from processing.dispersion_curves import read_max_file, read_txt_file
from obspy import read, Stream, UTCDateTime
import os


# DATA


def combine_mseed_files():
    # read from coordinate file
    data_path = "./data/WH03/2025_WHY_MTS/"
    instrument_names = [
        "C",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "XE1",
        "XE2",
        "XE3",
        "XE4",
        "XW1",
        "XW2",
        "XW3",
        "XW4",
    ]

    for inst in instrument_names:
        for coord in ["N", "E", "Z"]:
            traces = []
            for file in os.listdir(data_path):
                if inst not in file or "." + coord + "." not in file:
                    continue
                stream = read(data_path + file)
                traces.append(stream[0])
            output_fname = "./data/WH03/mseed_files/" + inst + coord + ".miniseed"
            combined = Stream(traces)
            combined.write(output_fname, format="MSEED")


def split_miniseed_components():
    sites = [
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
        "TP09",
        "TP10",
    ]

    file_names = ["./data/WH02_3C/" + s + "_WH02.mseed" for s in sites]

    for f in file_names:
        st = read(f)
        for s in st:
            chan = s.stats["channel"][-1]
            s.write(
                f.replace("WH02_3C", "WH02_3C_split").replace(
                    ".mseed", "" + chan + ".mseed"
                ),
                format="MSEED",
            )


def plot_raw_data():
    # data_path = "./data/WH03/2025_WHY_MTS/"
    data_path = "./data/WH04/2025_WHY_MTS/"
    for file in os.listdir(data_path):
        if ".miniseed" in file:
            stream = read(data_path + file)
            if stream[0].stats["channel"] == "EPZ":
                # 6:30 - 9:30
                # 00:48 - 13:17
                tr = stream[0]
                tr = tr.trim(
                    # longest section WH03
                    # starttime=UTCDateTime(2025, 10, 23, 18, 00, 00),
                    # endtime=UTCDateTime(2025, 10, 23, 21, 15, 00),
                    # longest section WH04
                    # starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                    # endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                    # quietest section WH04
                    starttime=UTCDateTime(2025, 10, 24, 8, 0, 0),
                    endtime=UTCDateTime(2025, 10, 24, 11, 0, 0),
                )
                plt.plot(tr)
                # tr.plot()
                # print(tr.stats)
    plt.show()


def slice_noise_data():
    # data_path = "./data/WH03/2025_WHY_MTS/"
    data_path = "./data/WH04/2025_WHY_MTS/"
    for file in os.listdir(data_path):
        if ".miniseed" in file:
            stream = read(data_path + file)
            # 6:30 - 9:30
            # 00:48 - 13:17
            tr = stream[0]
            tr = tr.trim(
                # longest section WH03
                # starttime=UTCDateTime(2025, 10, 23, 18, 00, 00),
                # endtime=UTCDateTime(2025, 10, 23, 21, 15, 00),
                # longest section WH04
                # starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                # endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                # quietest section WH04
                starttime=UTCDateTime(2025, 10, 24, 8, 0, 0),
                endtime=UTCDateTime(2025, 10, 24, 11, 0, 0),
            )

            output_fname = "./data/WH04/quietest_slice/" + file
            sliced_tr = Stream(tr)
            sliced_tr.write(output_fname, format="MSEED")


def ambient_noise_data(site):
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
        "TP09",
        "TP10",
    ]

    dir_path = "./data/" + site + "/mseeds/"
    for s in stations:
        st = read(dir_path + s + "_" + site + ".mseed")
        print(st[0].stats)


# MAPS


# AMBIENT NOISE


def plot_ambient_noise():
    pass


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
    plt.show()


# DISPERSION CURVES


def plot_computed_dispersion_curve(
    max_path, f_range, err_thresh=None, freq_outliers=[], vel_outliers=[]
):
    """
    Plot dispersion curve from max file.
    """
    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
        err_thresh=err_thresh,
        freq_outliers=freq_outliers,
        vel_outliers=vel_outliers,
    )

    inds = np.full(len(freqs), False)
    for f_min, f_max in f_range:
        # save the frequencies between frequency bounds
        inds = inds | (freqs >= f_min) & (freqs <= f_max)

    # freq_bins = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)), 75)
    # vel_bins = np.logspace(np.log10(np.min(vel_meds)), np.log10(np.max(vel_meds)), 100)
    freq_bins = np.logspace(
        np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs)
    )
    vel_bins = np.logspace(np.log10(np.min(vel_meds)), np.log10(np.max(vel_meds)), 75)

    plt.figure(figsize=(10, 6))
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

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.colorbar(label="counts")

    # plot dispersion curve with errors
    # plt.plot(freqs_curve, vels_curve)

    plt.errorbar(
        freqs[inds],
        vel_meds[inds],
        stds[inds],
        marker="o",
        markersize=3,
        c="black",
        elinewidth=1,
        barsabove=True,
    )

    plt.grid(True)

    plt.tight_layout()

    path = (
        "./figures/dispersion_curves/"
        + max_path.split("/")[-1].split("_fine.")[0]
        + "_curve.png"
    )
    # plt.savefig(path)
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

    # plt.ylim([190, 2100])

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

    path = (
        "./figures/dispersion_curves/"
        + max_path.split("/")[-1].split("_fine.")[0]
        + "_curve.png"
    )
    plt.savefig(path)


def compare_dispersion_curves(max_path, txt_path, f_min, f_max):
    plt.figure(figsize=(10, 6))

    df_max = read_max_file(max_path)

    freqs_grid, vels_grid, freqs, vel_meds, vel_means, stds, inds = (
        compute_dispersion_curve(df_max, f_min, f_max)
    )

    freq_bins = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)), 75)
    vel_bins = np.logspace(np.log10(np.min(vel_meds)), np.log10(np.max(vel_meds)), 100)

    # plt.subplot(1, 2, 1)
    # plot frequency and velocity 2D histogram
    plt.hist2d(
        freqs_grid,
        vels_grid,
        bins=[
            freq_bins,
            vel_bins,
        ],
        cmap="viridis",
        norm=LogNorm(),
        zorder=1,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.ylim([190, 2010])
    plt.xlim([2.0, 8.0])

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.colorbar(label="counts")

    # plot dispersion curve with errors

    plt.errorbar(
        freqs[inds],
        vel_means[inds],
        stds[inds],
        # marker="o",
        # linestyle=None,
        label="mean",
        c="red",
        elinewidth=3,
        barsabove=True,
        fmt="o",
        ms=5,
        zorder=2,
    )

    plt.scatter(
        freqs[inds],
        vel_meds[inds],
        # marker="o",
        # linestyle="-",
        label="median",
        c="yellow",
        s=50,
        zorder=3,
    )

    # txt file
    df_txt = read_txt_file(txt_path)

    plt.scatter(
        df_txt["frequency"],
        1 / df_txt["slowness"],
        # marker="o",
        # linestyle="-",
        label="geopsy",
        c="orange",
        s=50,
        zorder=3,
    )
    """
    plt.errorbar(
        df_txt["frequency"],
        1 / df_txt["slowness"],
        1 / (df_txt["percent_error"] * df_txt["slowness"]),
        #c="black",
        alpha=0.5,
    )
    """

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    """
    plt.subplot(1, 2, 2)
    plt.scatter(
        df_txt["frequency"],
        (1 / df_txt["slowness"]) - vel_means[inds],
        # marker="o",
        # linestyle="-",
        label="geopsy - mean",
    )
    plt.scatter(
        df_txt["frequency"],
        (1 / df_txt["slowness"]) - vel_meds[inds],
        # marker="o",
        # linestyle="-",
        label="geopsy - median",
    )
    plt.axhline(0)
    plt.legend()
    
    plt.tight_layout()
    """

    path = "./figures/compare-field-" + max_path.split("/")[-1].split(".")[0] + ".png"
    plt.savefig(path)


def plot_dispersion_curve_frequency(max_path, freq):
    """
    Plot dispersion curve from max file.
    """
    plt.clf()
    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max
    )

    plt.figure(figsize=(6, 4))

    freq_diff = np.abs(freqs - freq)
    ind = np.where(freq_diff == freq_diff.min())[0]
    grid_inds = np.where(freqs_grid == np.full(freqs_grid.shape, freqs[ind]))[0]

    plt.hist(vels_grid[grid_inds], bins=40)

    # plot frequency and velocity 2D histogram
    # plt.xscale("log")

    plt.xlabel("velocity (m/s)")
    plt.ylabel("counts")

    plt.axvline(vel_meds[ind], c="black")
    plt.axvline(vel_meds[ind] - stds[ind], c="red")
    plt.axvline(vel_meds[ind] + stds[ind], c="red")

    plt.title("freq: " + str(freq) + " Hz")
    plt.grid(True)

    plt.tight_layout()

    path = (
        "./figures/freqs/capon-WH01-test24/"
        + max_path.split("/")[-1]
        + "_freq "
        + str(freq)
        + ".png"
    )
    plt.savefig(path)
    plt.close()


def plot_paper_dispersion_curves():
    data_WH01, freqs_grid_WH01, vels_grid_WH01 = setup_data(site="WH01")
    data_WH02, freqs_grid_WH02, vels_grid_WH02 = setup_data(site="WH02")

    freqs_grid_WH01 = freqs_grid_WH01.values
    vels_grid_WH01 = vels_grid_WH01.values
    freqs_grid_WH02 = freqs_grid_WH02.values
    vels_grid_WH02 = vels_grid_WH02.values

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    freqs_WH01 = 1 / data_WH01.periods
    vels_WH01 = data_WH01.data_obs * 1000

    freqs_WH02 = 1 / data_WH02.periods
    vels_WH02 = data_WH02.data_obs * 1000

    freq_bins_WH01 = np.logspace(
        np.log10(np.min(freqs_grid_WH01)),
        np.log10(np.max(freqs_grid_WH01)),
        len(np.unique(freqs_grid_WH01)),
    )
    vel_bins_WH01 = np.logspace(
        np.log10(np.min(vels_grid_WH01)), np.log10(np.max(vels_grid_WH01)), 75
    )

    freq_bins_WH02 = np.logspace(
        np.log10(np.min(freqs_grid_WH02)),
        np.log10(np.max(freqs_grid_WH02)),
        len(np.unique(freqs_grid_WH02)),
    )
    vel_bins_WH02 = np.logspace(
        np.log10(np.min(vels_grid_WH02)), np.log10(np.max(vels_grid_WH02)), 75
    )

    # plot frequency and velocity 2D histogram
    h1 = ax1.hist2d(
        freqs_grid_WH01,
        vels_grid_WH01,
        bins=[
            freq_bins_WH01,
            vel_bins_WH01,
        ],
        cmap="coolwarm",
        norm=LogNorm(),
    )
    h2 = ax2.hist2d(
        freqs_grid_WH02,
        vels_grid_WH02,
        bins=[
            freq_bins_WH02,
            vel_bins_WH02,
        ],
        cmap="coolwarm",
        norm=LogNorm(),
    )

    ax1.errorbar(
        freqs_WH01,
        vels_WH01,
        data_WH01.sigma_data * 1000,
        c="black",
        # alpha=0.5,
        elinewidth=1,
        capsize=2,
        ls="none",
        fmt="o",
        markersize=2,
    )
    ax2.errorbar(
        freqs_WH02,
        vels_WH02,
        data_WH02.sigma_data * 1000,
        c="black",
        # alpha=0.5,
        elinewidth=1,
        capsize=2,
        ls="none",
        fmt="o",
        markersize=2,
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Phase velocity (m/s)")
    ax1.grid(True)

    ax2.set_xlabel("Frequency (Hz)")
    # ax2.set_ylabel("Phase velocity (m/s)")

    x_ticks = [1, 2, 3, 4, 5, 7, 10, 15]
    y_ticks = [200, 300, 400, 600, 1000, 2000]

    """
    [ 1.11440654, 14.59924331]
    [ 201.02538911, 1999.34424147]
    [ 1., 14.59924331]
    [ 201.02734988, 1998.29528439]
    """
    ax1.set_xticks(x_ticks)
    ax2.set_xticks(x_ticks)
    # ax2.set_xticks(minor_ticks, minor=True)

    ax1.set_yticks(y_ticks)
    # ax1.set_yticks(minor_ticks, minor=True)
    ax2.set_yticks(y_ticks)
    # ax2.set_yticks(minor_ticks, minor=True)

    # ax2.grid(True, which="both")
    ax1.grid(which="major", alpha=0.75)
    ax1.grid(which="minor", alpha=0.5)

    ax2.grid(which="major", alpha=0.75)
    ax2.grid(which="minor", alpha=0.5)

    ax1.text(1.03, 1800, "a)")
    ax2.text(1.03, 1800, "b)")

    for axis in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.tight_layout()

    cbaxes = inset_axes(
        ax2,
        width="5%",
        height="35%",
        loc="upper right",
        bbox_to_anchor=(-0.02, 0.05, 1.0, 1.0),
        bbox_transform=ax2.transAxes,
        borderpad=2.5,
    )
    cbar = fig.colorbar(h2[3], ax=[ax1, ax2], cax=cbaxes, pad=0.5)
    # plt.colorbar(cax=cbaxes, ticks=[0.0, 1], orientation="horizontal")
    cbar.set_label("Counts", labelpad=-55)
    cbar.ax.set_yticks([1, 10, 100])
    cbar.ax.set_yticklabels([1, 10, 100])

    # path = "./figures/final/dispersion_curves.png"
    path = "./figures/final/dispersion_curves.pdf"
    fig.savefig(path, dpi=600)


def plot_full_results(input_ds, results_ds, n_bins=100, save=False, out_filename=""):
    """
    plot results for paper.
    subplots
    - site location
    - data pred vs. data obs
    - depth profile (near surface)
    """

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

    freqs = 1 / input_ds["period"]

    yerr = input_ds.attrs["sigma_data"]

    # get data prediction
    pred_ind = np.argmax(results_ds["logL"].values)

    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    # read in site locations as png?

    # Create a figure
    fig = plt.figure(figsize=(10, 6))

    # Define a GridSpec layout
    gs = GridSpec(2, 2, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0, 0])  # site location
    ax2 = fig.add_subplot(gs[1, 0])  # data pred vs. obs
    ax3 = fig.add_subplot(gs[:, 1])  # depth profile

    # PLOT SITE LOCATION
    # reading png image file
    path = (
        "/home/jbyer/Documents/uoc/repos/mapping/Get_Site_Locations_Jamie/WH01_map.png"
    )
    # path = "/home/jbyer/Documents/uoc/repos/mapping/Get_Site_Locations_Jamie/WH02_map.png"
    im = img.imread(path)
    # show image
    ax1.imshow(im)
    ax1.axis("off")

    # PLOT DATA
    ax2.hist2d(hist_freqs, data_preds, bins=n_bins, cmin=1, norm="log")
    # fig.colorbar(im, ax=ax, label="count")
    ax2.scatter(
        freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred"
    )
    ax2.errorbar(
        freqs,
        input_ds["data_obs"],
        yerr,
        fmt="o",
        zorder=3,
        c="orange",
        label="data_obs",
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("velocity (km/s)")

    # ax2.legend()

    # PLOT DEPTH PROFILE
    # use results_ds to get model params
    model_params = results_ds["model_params"].values
    # define hist bins between bounds
    # use param inds to get depth, and use min and max of all depth bounds
    depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]
    depth_bins = (
        np.linspace(
            np.min(depth_bounds[:, 0]),
            np.max(depth_bounds[:, 1]),
            n_bins,
        )
        * 1000
    )  # unit conversion
    vel_s_bounds = input_ds["param_bounds"][input_ds["vel_s_inds"]]
    vel_s_bins = np.linspace(
        np.min(vel_s_bounds[:, 0]), np.max(vel_s_bounds[:, 1]), n_bins
    )
    counts = np.zeros((n_bins, n_bins))

    # loop over every resulting model
    # add vel_s 1 to hist bins above depth
    # add vel_s 2 to hist bins below depth

    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    n_steps = len(results_ds["step"])

    depth = model_params[depth_inds] * 1000  # unit conversion to m
    depth_plotting = np.concatenate(
        (
            np.zeros((1, n_steps)),
            depth,
            np.full((1, n_steps), np.max(depth_bounds[:, 1])) * 1000,  # unit conversion
        ),
        axis=0,
    )
    vel_s = model_params[vel_s_inds]

    # for each layer
    # for each sample / step
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        for step_ind in range(n_steps):
            # find bin index closest to layer depth
            depth_upper_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind, step_ind])
            )
            depth_lower_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind + 1, step_ind])
            )
            # find bin index closest to layer vel_s
            vel_s_close_inds = np.argmin(abs(vel_s_bins - vel_s[layer_ind, step_ind]))

            counts[depth_upper_inds:depth_lower_inds, vel_s_close_inds] += 1

    h = ax3.imshow(
        counts,
        norm=LogNorm(),
        extent=[vel_s_bins[0], vel_s_bins[-1], depth_bins[-1], depth_bins[0]],
        aspect="auto",
        interpolation="none",
    )
    ax3.set_ylabel("depth (m)")

    ax3.set_xlim(ax1.get_xlim()[::-1])
    plt.gca().invert_yaxis()

    if save:
        plt.savefig("figures/" + out_filename + "/results-" + out_filename + ".png")
    else:
        plt.show()


def plot_computed_dispersion_curve_curr(max_path):
    """
    Plot dispersion curve from max file.
    """
    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    vel_bins = np.logspace(
        np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), len(vel_meds) + 1
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))

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
    # plt.ylim([100, 2200])
    plt.ylim([50, 2200])

    plt.xlabel("frequency (Hz)")
    plt.ylabel("phase velocity (m/s)")

    plt.colorbar(label="counts")

    # plot dispersion curve with errors
    # plt.plot(freqs_curve, vels_curve)

    """
    plt.errorbar(
        freqs,
        vel_meds,
        stds,
        marker="o",
        markersize=3,
        c="black",
        elinewidth=1,
        barsabove=True,
    )
    """
    for axis in [ax1.xaxis, ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.grid(True)

    plt.title(
        "\nPERIOD_COUNT=20, WINDOW_OVERLAP (%)=30, ANTI-TRIGGERING_ON_RAW_SIGNAL (y/n)=y"
        "\nSTATISTIC_COUNT=500, FREQ_BAND_WIDTH=0.10"
        "\nGRID_STEP (rad/m)= 0.005, GRID_SIZE (rad/m)=0.9, N_MAXIMA=1"
    )
    plt.tight_layout()

    path = "./figures/WH04/1C/conventional-WH04-longest-test02.png"
    # path = "./figures/WH04/2C/conventionaltransverse-WH04-longest-default02.png"
    # path = "./figures/WH04/3C/rtbf-WH04-longest-test01.png"
    plt.savefig(path)
    # plt.show()


def plot_double_dispersion_curves(max_paths):
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    titles = ["Vertical", "Transverse"]

    for ind, p in enumerate(max_paths):
        df_max = read_max_file(p)
        freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = (
            compute_dispersion_curve(
                df_max,
            )
        )

        freq_bins = np.logspace(
            np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
        )
        vel_bins = np.logspace(
            np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), len(vel_meds) + 1
        )

        # plot frequency and velocity 2D histogram
        axes[ind].hist2d(
            freqs_grid,
            vels_grid,
            bins=[
                freq_bins,
                vel_bins,
            ],
            cmap="coolwarm",
            norm=LogNorm(),
        )

        axes[ind].set_xscale("log")
        axes[ind].set_yscale("log")
        # plt.ylim([100, 2200])
        axes[ind].set_ylim([50, 2200])

        axes[ind].set_xlabel("frequency (Hz)")
        axes[ind].set_ylabel("phase velocity (m/s)")

        # axes[ind].colorbar(label="counts")

        # plot dispersion curve with errors
        # plt.plot(freqs_curve, vels_curve)

        """
        axes[ind].errorbar(
            freqs,
            vel_meds,
            stds,
            marker="o",
            markersize=3,
            c="black",
            elinewidth=1,
            barsabove=True,
        )
        """
        for axis in [axes[ind].xaxis, axes[ind].yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)

        axes[ind].grid(True)
        axes[ind].grid(True, which="minor", alpha=0.5)
        axes[ind].set_title(titles[ind])

    plt.suptitle("WH04")
    plt.tight_layout()

    # path = "./figures/processing/WH02-default02.png"
    # path = "./figures/sensitivity_tests/threshold/conventional-WH01-test10.png"
    # path = "./figures/processing/WH01-3C-test02.png"
    # plt.savefig(path)
    plt.show()


def plot_multiple_dispersion_curves(max_paths):
    """
    Plot dispersion curve from max file.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    for p in max_paths:
        df_max = read_max_file(p)
        freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = (
            compute_dispersion_curve(
                df_max,
            )
        )

        freq_bins = np.logspace(
            np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs)
        )
        vel_bins = np.logspace(
            np.log10(np.min(vel_meds)), np.log10(np.max(vel_meds)), len(vel_meds)
        )

        # plot frequency and velocity 2D histogram
        """
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
        """

        # plot dispersion curve with errors
        # plt.plot(freqs_curve, vels_curve)

        plt.errorbar(
            freqs,
            vel_meds,
            stds,
            marker="o",
            markersize=3,
            # c="black",
            elinewidth=1,
            barsabove=True,
        )

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.xscale("log")
    plt.yscale("log")

    for axis in [ax1.xaxis, ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.grid(True)

    plt.tight_layout()

    # plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    # max_path = "./results/WH01/conventional-WH01-test02.max"
    # plot_computed_dispersion_curve_curr(max_path)
    pass
