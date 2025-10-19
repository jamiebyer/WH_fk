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


def compute_dispersion_curve(df, err_thresh):
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
        vel_med = np.median(vels_grid[freqs_grid == f])
        vel_mean = np.mean(vels_grid[freqs_grid == f])
        std = np.std(vels_grid[freqs_grid == f])
        vel_meds_curve.append(vel_med)
        vel_means_curve.append(vel_mean)
        stds_curve.append(std)

    # error threshold
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


def get_profile():
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    if plot_true_model:
        true_params = input_ds["model_true"].values

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

    # plot true model overtop
    if plot_true_model:
        true_depth = true_params[depth_inds] * 1000
        true_vel_s = true_params[vel_s_inds]
        true_depth_plotting = np.concatenate(
            ([0], true_depth, [np.max(depth_bounds[:, 1]) * 1000])
        )

        true_model = []
        for layer_ind in range(input_ds.attrs["n_layers"] + 1):
            true_model.append([true_depth_plotting[layer_ind], true_vel_s[layer_ind]])
            true_model.append(
                [true_depth_plotting[layer_ind + 1], true_vel_s[layer_ind]]
            )

    fig = plt.figure()
    gs = GridSpec(1, 3, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:], sharey=ax1)

    # plot depth histogram
    for ind in range(input_ds.attrs["n_layers"]):
        ax1.hist(
            depth[ind],
            bins=depth_bins,
            density=True,
            orientation="horizontal",
        )

    ax1.set_ylim(
        [
            np.min(depth_bounds[:, 0]) * 1000,
            np.max(depth_bounds[:, 1]) * 1000,
        ]
    )
    ax1.set_ylabel("depth (m)")

    ax1.set_xlim(ax1.get_xlim()[::-1])
    plt.gca().invert_yaxis()

    h = ax2.imshow(
        counts,
        norm=LogNorm(),
        extent=[vel_s_bins[0], vel_s_bins[-1], depth_bins[-1], depth_bins[0]],
        aspect="auto",
        interpolation="none",
    )

    # plot true model overtop
    if plot_true_model:
        true_model = np.array(true_model)
        ax2.plot(true_model[:, 1], true_model[:, 0], c="red")

    fig.colorbar(h, ax=ax2)
    ax2.set_xlabel("vel s (km/s)")

    # make these tick labels invisible
    ax2.tick_params("y", labelleft=False)

    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/profile-" + out_filename + ".png")
    else:
        plt.show()


def compute_vs30():
    # time-averaged shear-wave velocity to 30 m depth

    # read in inversion results
    # get most probable model params

    # compute vs30
    # sum the depths
    # determine what depth contains 30m, use params until that point
    # interface at 30m

    vs30 = np.sum(depth) / np.sum(depth / vel_s)
