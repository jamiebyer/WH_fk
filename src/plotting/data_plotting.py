import numpy as np
import matplotlib.pyplot as plt

from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import (
    array_transff_wavenumber,
    array_transff_freqslowness,
)

from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

import matplotlib.image as img
import xarray as xr

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
import obspy
from obspy import read, Stream, UTCDateTime
import os


# DATA


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
        # "slowness",
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


def plot_full_results(
    input_ds, results_ds, site, n_bins=100, save=False, out_filename=""
):
    """
    plot results for paper.
    subplots
    - site location
    - data pred vs. data obs
    - depth profile (near surface)
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)
    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # read in site locations as png?

    # Create a figure
    fig = plt.figure(figsize=(20, 10))

    # Define a GridSpec layout
    gs = GridSpec(4, 5, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # site location
    ax2 = fig.add_subplot(gs[2:, 0:2])  # data pred vs. obs
    ax3 = fig.add_subplot(gs[:, 2:-1])  # depth vel_s profile
    ax4 = fig.add_subplot(gs[:, -1])  # depth hist profile

    # PLOT SITE LOCATION
    # reading png image file
    path = "./results/maps/" + site + "_map.png"
    # path = "/home/jbyer/Documents/uoc/repos/mapping/Get_Site_Locations_Jamie/WH02_map.png"
    im = img.imread(path)
    # show image
    ax1.imshow(im)
    ax1.axis("off")

    # PLOT DATA
    freqs = 1 / input_ds["period"]
    yerr = input_ds.attrs["sigma_data"]

    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten() * 1000

    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    bin_freqs = np.flip(freqs)
    logbins = np.logspace(np.log10(2), np.log10(13), len(freqs) + 1)

    # make freq bins from full grid
    max_path = "./data/" + site + "/max_files/" + site + "_fine.max"
    df_max = read_max_file(max_path)
    freqs_grid = df_max["frequency"].values
    vels_grid = 1 / df_max["slowness"].values

    freq_bins = np.unique(freqs_grid)
    vels_bins = np.unique(vels_grid)

    ax2.hist2d(hist_freqs, data_preds, bins=[freq_bins, vels_bins], cmin=1, norm="log")
    # fig.colorbar(im, ax=ax, label="count")
    ax2.errorbar(
        freqs,
        input_ds["data_obs"] * 1000,
        yerr * 1000,
        fmt="o",
        zorder=3,
        c="black",
        markersize=3,
        label="data_obs",
    )

    # 1.1144065354855293 14.599243307112378
    # 1.0 14.599243307112378
    # 201.0253891072783 1999.3442414661472
    # 201.02734987527654 1998.2952843920082

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (Hz)", fontsize=20)
    ax2.set_ylabel("Phase velocity (m/s)", fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=18)

    for axis in [ax2.xaxis, ax2.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    ax2.set_xlim([2, 14.6])
    ax2.set_ylim([200, 2000])

    x_ticks = [2, 3, 4, 5, 6, 7, 10, 15]
    y_ticks = [200, 300, 400, 600, 1000, 2000]

    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)

    ax2.grid(which="major", alpha=0.75)
    ax2.grid(which="minor", alpha=0.5)

    # PLOT DEPTH PROFILE
    # use results_ds to get model params
    model_params = results_ds["model_params"].values
    # define hist bins between bounds
    # use param inds to get depth, and use min and max of all depth bounds
    depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]
    depth_bins = (
        np.linspace(
            # np.min(depth_bounds[:, 0]),
            0,
            np.max(depth_bounds[:, 1]),
            n_bins,
        )
        * 1000
    )  # unit conversion
    vel_s_bounds = input_ds["param_bounds"][input_ds["vel_s_inds"]] * 1000
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
    vel_s = model_params[vel_s_inds] * 1000

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

    ax3.set_xlim([200, 2000])
    ax3.set_ylim([0, 125])

    ax3.set_xticks(np.arange(200, 2001, 400))
    ax3.set_xticks(np.arange(200, 2001, 50), minor=True)
    ax3.set_yticks([0, 20, 40, 60, 80, 100, 120])
    ax3.set_yticks(np.arange(0, 120, 5), minor=True)
    ax3.grid()
    ax3.grid(which="minor", alpha=0.3)

    ax3.set_ylim(ax3.get_ylim()[::-1])

    ax3.tick_params("y", labelleft=False)
    ax3.yaxis.tick_right()

    ax3.set_xlabel("Shear velocity (m/s)", fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=18)

    # plot depth histogram
    for ind in range(input_ds.attrs["n_layers"]):
        ax4.hist(
            depth[ind],
            bins=depth_bins,
            density=True,
            orientation="horizontal",
        )

    ax4.set_xlabel("Probability", fontsize=20)
    ax4.set_ylabel("Depth (m)", fontsize=20)
    ax4.yaxis.set_label_position("right")

    ax4.set_ylim([0, 125])
    ax4.set_xticks([0.0, 0.25, 0.5])
    ax4.set_xticks(np.arange(0, 0.576, 0.025), minor=True)
    ax4.set_yticks([0, 20, 40, 60, 80, 100, 120])
    ax4.set_yticks(np.arange(0, 120, 5), minor=True)
    ax4.tick_params(axis="both", which="major", labelsize=18)
    ax4.yaxis.tick_right()
    ax4.grid()
    ax4.grid(which="minor", alpha=0.3)

    ax4.set_ylim(ax4.get_ylim()[::-1])

    ax1.text(1500, 0.5, "a)", c="k", fontsize=20)
    ax2.text(13, 1600, "b)", fontsize=20)
    ax3.text(240, 122, "c)", fontsize=20)
    ax4.text(0.03, 122, "d)", fontsize=20)

    if save:
        plt.savefig(
            "figures/" + out_filename + "/results-" + out_filename + ".pdf", dpi=600
        )
    else:
        plt.show()


def plot_vs30(file_names):
    """
    Vs30 = sum(d_i)/sum(t_i) = 30/sum(d_i/v_i)

    Description	VS30 range (m/s)
    Hard rock	1500
    Rock	760-1500
    Very dense soil and soft rock	360-760
    Stiff soil	180-360
    Soil with soft clay	<180
    Site-specific analysis required	---
    """
    for file_name in file_names:
        input_path = "./results/inversion/input-" + file_name + ".nc"
        results_path = "./results/inversion/results-" + file_name + ".nc"

        input_ds = xr.open_dataset(input_path)
        results_ds = xr.open_dataset(results_path)

        # n_burn = input_ds.attrs["n_burn"]
        n_burn = int(len(results_ds["step"]) / 3)

        # cut results by step
        results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

        # use results_ds to get model params
        model_params = results_ds["model_params"].values

        depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]

        depth_inds = input_ds["depth_inds"]
        vel_s_inds = input_ds["vel_s_inds"]

        n_steps = len(results_ds["step"])

        depth = model_params[depth_inds] * 1000  # unit conversion to m
        depth_plotting = np.concatenate(
            (
                np.zeros((1, n_steps)),
                depth,
                np.full((1, n_steps), np.max(depth_bounds[:, 1]))
                * 1000,  # unit conversion
            ),
            axis=0,
        )

        vel_s = model_params[vel_s_inds]

        # depth_boundary = 10
        depth_boundary = 30
        Vs30_list = []
        # for each layer
        # for each sample / step
        for step_ind in range(n_steps):
            # find first depth after 30 m
            depth_diff = depth_plotting[:, step_ind] - depth_boundary
            depth_diff[depth_diff < 0] = np.inf

            # smallest positive number
            layer_ind = np.argmin(depth_diff)
            depth_plotting[layer_ind] = depth_boundary

            thickness = (
                depth_plotting[1 : layer_ind + 1, step_ind]
                - depth_plotting[:layer_ind, step_ind]
            )

            Vs30 = depth_boundary / np.sum(
                thickness[: layer_ind + 1] / vel_s[:layer_ind, step_ind]
            )
            Vs30_list.append(Vs30)

        plt.hist(np.array(Vs30_list) * 1000, bins=30, density=True, alpha=0.5)

    classes = [
        # ["A", "Hard\nrock", 1500, 1550],
        # ["B", "Rock", 760, 900],
        ["C", "C (Very dense soil / soft rock)", 360, 380],
        ["D", "D (Stiff soil)", 180, 230],
        ["E", "E (Soft soil)", 0, 0],
    ]
    for c, name, vert, loc in classes:
        # plt.text(loc, 0.15, name)
        plt.axvline(vert, c="k", ls="-")

    # add percentage for classification

    plt.xlabel("Vs30 (m/s)")
    # plt.xlabel("Vs10 (m/s)")
    plt.ylabel("Probability")

    # plt.xlim([0, 760])
    plt.xlim([100, 450])

    plt.xticks(np.arange(100, 450, 50))
    plt.xticks(np.arange(100, 450, 10), minor=True)
    plt.yticks(np.arange(0, 0.16, 0.04))
    plt.yticks(np.arange(0, 0.16, 0.01), minor=True)
    plt.grid()
    plt.grid(which="minor", alpha=0.3)

    plt.tight_layout()

    # plt.savefig("figures/vs10.png")
    plt.savefig("figures/vs30.png")


def plot_vs30_subplots(file_names):
    """
    Vs30 = sum(d_i)/sum(t_i) = 30/sum(d_i/v_i)

    Description	VS30 range (m/s)
    Hard rock	1500
    Rock	760-1500
    Very dense soil and soft rock	360-760
    Stiff soil	180-360
    Soil with soft clay	<180
    Site-specific analysis required	---
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for file_name in file_names:
        input_path = "./results/inversion/input-" + file_name + ".nc"
        results_path = "./results/inversion/results-" + file_name + ".nc"

        input_ds = xr.open_dataset(input_path)
        results_ds = xr.open_dataset(results_path)

        # n_burn = input_ds.attrs["n_burn"]
        n_burn = int(len(results_ds["step"]) / 3)

        # cut results by step
        results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

        # use results_ds to get model params
        model_params = results_ds["model_params"].values

        depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]

        depth_inds = input_ds["depth_inds"]
        vel_s_inds = input_ds["vel_s_inds"]

        n_steps = len(results_ds["step"])

        depth = model_params[depth_inds] * 1000  # unit conversion to m
        depth_plotting = np.concatenate(
            (
                np.zeros((1, n_steps)),
                depth,
                np.full((1, n_steps), np.max(depth_bounds[:, 1]))
                * 1000,  # unit conversion
            ),
            axis=0,
        )

        vel_s = model_params[vel_s_inds]

        depth_boundary = 10
        Vs10_list = []
        # for each layer
        # for each sample / step
        for step_ind in range(n_steps):
            # find first depth after 30 m
            depth_diff = depth_plotting[:, step_ind] - depth_boundary
            depth_diff[depth_diff < 0] = np.inf

            # smallest positive number
            layer_ind = np.argmin(depth_diff)
            depth_plotting[layer_ind] = depth_boundary

            thickness = (
                depth_plotting[1 : layer_ind + 1, step_ind]
                - depth_plotting[:layer_ind, step_ind]
            )

            Vs10 = depth_boundary / np.sum(
                thickness[: layer_ind + 1] / vel_s[:layer_ind, step_ind]
            )
            Vs10_list.append(Vs10)

        ax[0].hist(np.array(Vs10_list) * 1000, bins=10, density=True, alpha=0.5)

    for file_name in file_names:
        input_path = "./results/inversion/input-" + file_name + ".nc"
        results_path = "./results/inversion/results-" + file_name + ".nc"

        input_ds = xr.open_dataset(input_path)
        results_ds = xr.open_dataset(results_path)

        # n_burn = input_ds.attrs["n_burn"]
        n_burn = int(len(results_ds["step"]) / 3)

        # cut results by step
        results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

        # use results_ds to get model params
        model_params = results_ds["model_params"].values

        depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]

        depth_inds = input_ds["depth_inds"]
        vel_s_inds = input_ds["vel_s_inds"]

        n_steps = len(results_ds["step"])

        depth = model_params[depth_inds] * 1000  # unit conversion to m
        depth_plotting = np.concatenate(
            (
                np.zeros((1, n_steps)),
                depth,
                np.full((1, n_steps), np.max(depth_bounds[:, 1]))
                * 1000,  # unit conversion
            ),
            axis=0,
        )

        vel_s = model_params[vel_s_inds]

        depth_boundary = 30
        Vs30_list = []
        # for each layer
        # for each sample / step
        for step_ind in range(n_steps):
            # find first depth after 30 m
            depth_diff = depth_plotting[:, step_ind] - depth_boundary
            depth_diff[depth_diff < 0] = np.inf

            # smallest positive number
            layer_ind = np.argmin(depth_diff)
            depth_plotting[layer_ind] = depth_boundary

            thickness = (
                depth_plotting[1 : layer_ind + 1, step_ind]
                - depth_plotting[:layer_ind, step_ind]
            )

            Vs30 = depth_boundary / np.sum(
                thickness[: layer_ind + 1] / vel_s[:layer_ind, step_ind]
            )
            Vs30_list.append(Vs30)

        ax[1].hist(np.array(Vs30_list) * 1000, bins=10, density=True, alpha=0.5)

    classes = [
        # ["A", "Hard\nrock", 1500, 1550],
        # ["B", "Rock", 760, 900],
        ["C", "C (Very dense soil / soft rock)", 360, 380],
        ["D", "D (Stiff soil)", 180, 230],
        ["E", "E (Soft soil)", 0, 0],
    ]
    for c, name, vert, loc in classes:
        # plt.text(loc, 0.15, name)
        ax[0].axvline(vert, c="k", ls="-")
        ax[1].axvline(vert, c="k", ls="-")

    # add percentage for classification

    ax[0].set_xlabel("Vs10 (m/s)")
    ax[0].set_ylabel("Probability")

    ax[1].set_xlabel("Vs30 (m/s)")
    plt.setp(ax[1].get_yticklabels(), visible=False)
    # ax[1].set_ylabel("Probability")

    # plt.xlim([0, 760])
    ax[0].set_xlim([100, 450])
    ax[1].set_xlim([100, 450])

    ax[0].set_xticks(np.arange(100, 450, 50))
    ax[0].set_xticks(np.arange(100, 450, 10), minor=True)
    ax[0].set_yticks(np.arange(0, 0.16, 0.04))
    ax[0].set_yticks(np.arange(0, 0.16, 0.01), minor=True)
    ax[0].grid(alpha=0.5)
    ax[0].grid(which="minor", alpha=0.2)

    ax[1].set_xticks(np.arange(100, 450, 50))
    ax[1].set_xticks(np.arange(100, 450, 10), minor=True)
    ax[1].set_yticks(np.arange(0, 0.16, 0.04))
    ax[1].set_yticks(np.arange(0, 0.16, 0.01), minor=True)
    ax[1].grid(alpha=0.5)
    ax[1].grid(which="minor", alpha=0.2)

    ax[0].text(110, 0.142, "a)")
    ax[1].text(110, 0.142, "b)")

    plt.tight_layout()

    plt.savefig("figures/vs30_subplots.pdf")


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
        "\nPERIOD_COUNT=20, WINDOW_OVERLAP (%)=50, ANTI-TRIGGERING_ON_RAW_SIGNAL (y/n)=n"
        "\nSTATISTIC_COUNT=0, FREQ_BAND_WIDTH=0.10"
        "\nGRID_STEP (rad/m)= 0.005, GRID_SIZE (rad/m)=2.00, N_MAXIMA=0"
    )
    plt.tight_layout()

    path = "./figures/WH02/1C/conventional-WH02-default08.png"
    # path = "./figures/WH04/2C/conventionaltransverse-WH04-longest-default03.png"
    # path = "./figures/WH01/3C/rtbf-WH01-test01.png"
    plt.savefig(path)
    # plt.show()


def plot_slowness(max_path, curve_path):
    """
    Plot dispersion curve from max file.
    """
    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    slow_grid = np.log(1 / vels_grid)

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    # vel_bins = np.logspace(
    #     np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), len(vel_meds) + 1
    # )
    # slowness_bins = np.logspace(
    #     np.log10(np.min(1 / vels_grid)),
    #     np.log10(np.max(1 / vels_grid)),
    #     len(vel_meds) + 1,
    # )
    slowness_bins = np.linspace(np.min(slow_grid), np.max(slow_grid), len(freqs) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # plot frequency and velocity 2D histogram
    # """
    plt.hist2d(
        freqs_grid,
        np.log(1 / vels_grid),
        bins=[
            freq_bins,
            slowness_bins,
        ],
        cmap="coolwarm",
        norm=LogNorm(),
    )
    # """

    df = pd.read_csv(curve_path)

    percent_err = 100 * (df["stds"] / df["vels"])
    curve = np.log(1 / df["vels"])
    new_err = np.abs((percent_err / 100) * curve)

    plt.errorbar(
        df["freqs"],
        curve,
        # yerr=new_err,
        marker="o",
        markersize=2,
        c="black",
    )

    plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim([100, 2200])
    # plt.ylim([50, 2200])
    # plt.ylim([0, 0.0220])

    plt.xlabel("frequency (Hz)")
    plt.ylabel("ln(slowness)")

    # plt.colorbar(label="counts")

    # plot dispersion curve with errors
    # plt.plot(freqs_curve, vels_curve)

    for axis in [ax1.xaxis, ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.grid(True)

    # plt.title(
    #    "\nPERIOD_COUNT=20, WINDOW_OVERLAP (%)=50, ANTI-TRIGGERING_ON_RAW_SIGNAL (y/n)=n"
    #    "\nSTATISTIC_COUNT=0, FREQ_BAND_WIDTH=0.10"
    #    "\nGRID_STEP (rad/m)= 0.005, GRID_SIZE (rad/m)=2.00, N_MAXIMA=0"
    # )
    plt.tight_layout()

    path = "./figures/WH01/1C/slowness/conventional-WH01-log.png"
    # path = "./figures/WH02/1C/conventional-WH02-default08.png"
    # path = "./figures/WH04/2C/conventionaltransverse-WH04-longest-default03.png"
    # path = "./figures/WH01/3C/rtbf-WH01-test01.png"
    # plt.savefig(path)
    plt.show()


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


def plot_curve_picking():
    paths = [
        # "./results/curves/curve-WH01-1C.csv",
        # "./results/curves/curve-WH02-1C.csv",
        # "./results/curves/curve-WH03-1C.csv",
        # "./results/curves/curve-WH04-1C.csv",
        "./results/curves/curve-WH01-2C.csv",
        "./results/curves/curve-WH02-2C.csv",
        "./results/curves/curve-WH03-2C.csv",
        "./results/curves/curve-WH04-2C.csv",
    ]

    for p in paths:
        df = pd.read_csv(p)

        plt.errorbar(df["freqs"], df["vels"], yerr=df["stds"])

    # df = pd.read_csv("./results/curves/curve-WH02-1C-2.csv")
    # plt.errorbar(df["freqs"], df["vels"], yerr=df["stds"], c="blue")

    plt.xlabel("frequency (Hz)")
    plt.ylabel("velocity (m/s)")

    plt.xlim([1, 50])
    plt.ylim([50, 2200])
    plt.xscale("log")
    plt.yscale("log")

    # plt.legend(["WH01-1C", "WH02-1C", "WH03-1C", "WH04-1C"])
    plt.legend(["WH01-2C", "WH02-2C", "WH03-2C", "WH04-2C"])
    # plt.legend(["WH04-1C", "WH04-2C"])
    plt.show()


if __name__ == "__main__":
    # max_path = "./results/WH01/conventional-WH01-test02.max"
    # plot_computed_dispersion_curve_curr(max_path)
    pass
