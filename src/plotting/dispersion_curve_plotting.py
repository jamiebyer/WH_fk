import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

import shapely
from shapely import Point, Polygon

from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm

from plotting.distribution_fitting_plotting import plot_scaling_parameters
from utils.utils import read_max_file, get_path, get_k_limits, subset_data

# DISPERSION CURVES


def plot_computed_dispersion_curve(
    site, y_max, plot_polygon=False, plot_k_limits=False, fig=None, ax=None
):
    """
    Plot dispersion curve from max file.
    """
    max_path, curve_path, polygon_path = get_path(site)

    df_max = read_max_file(max_path)
    freqs = np.unique(df_max["frequency"].values)
    freqs_grid = df_max["frequency"].values
    vels_grid = 1 / df_max["slowness"].values

    n_bins = 200

    curve_df = pd.read_csv(curve_path)

    if plot_polygon:
        with open(polygon_path) as f:
            contents = f.read()
        # polygon = contents.replace("[", "").replace("]", "").split("), (")
        polygon = ast.literal_eval(contents)

    if plot_k_limits:
        k_min, k_max = get_k_limits(site)
        print(k_min, k_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 16))

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    # vel_bins = np.logspace(
    #     np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), n_bins
    # )
    # vel_bins = np.linspace(np.min(vels_grid), np.max(vels_grid), n_bins)
    # vel_bins = np.linspace(50, 1250, n_bins)
    # vel_bins = np.linspace(0, 1250, n_bins)
    vel_bins = np.linspace(0, y_max, n_bins)

    # plot frequency and velocity 2D histogram
    h = ax.hist2d(
        freqs_grid,
        vels_grid,
        bins=[
            freq_bins,
            vel_bins,
        ],
        # cmap="coolwarm",
        # norm=LogNorm(),
        cmin=1,
    )

    if plot_polygon:
        x, y = Polygon(polygon).exterior.xy
        ax.plot(x, y, c="red")

    ax.set_xscale("log")
    # plt.yscale("log")
    ax.set_ylim([50, y_max])
    # ax.set_ylim([50, 1250])

    # ax.set_ylim([0, 1250])

    # ax.set_ylabel("phase velocity (m/s)", fontsize=20)
    ax.set_ylabel("phase velocity (m/s)")

    # plt.colorbar(label="counts")

    ax.scatter(curve_df["freqs"], curve_df["vels"], s=10, c="black", edgecolor="white")

    # plot wavenumber limits
    if plot_k_limits:
        # k = 2*pi*f / v_p
        # v_1 = 2*pi*f/k
        v1 = 2 * np.pi * curve_df["freqs"] / k_min
        v2 = 2 * np.pi * curve_df["freqs"] / k_max
        v3 = 2 * np.pi * curve_df["freqs"] / (k_min / 2)
        v4 = 2 * np.pi * curve_df["freqs"] / (k_max / 2)
        ax.plot(curve_df["freqs"], v1, c="black")
        ax.plot(curve_df["freqs"], v2, c="black")
        ax.plot(curve_df["freqs"], v3, c="black", ls="dashed")
        ax.plot(curve_df["freqs"], v4, c="black", ls="dashed")

    # ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="major")

    # cbar = fig.colorbar(h[3], ax=ax)
    # cbar.set_ticks([])
    # cbar.set_label("counts")
    # cbar.ax.tick_params(labelsize=12)

    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    # plt.grid(True)

    # plt.show()

    return ax


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


# PLOT RESIDUALS


def plot_residuals(site, subset_type, y_max, remove_artifact, fig=None, ax=None):
    """
    plot the histogram of f-k beamforming results with the selected dispersion curve subtracted.
    read in the polygon that was selected with the polygon picker.
    plot the 5th and 95th quantiles of the data. Plot the same quantiles for the error distribution fit.
    """
    # with and without polygon slicing

    max_path, curve_path, polygon_path = get_path(site)

    df_max = read_max_file(max_path)
    freqs = np.unique(df_max["frequency"])
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]

    curve_df = pd.read_csv(curve_path)

    # if callback_context is None:
    with open(polygon_path) as f:
        contents = f.read()
    # polygon = contents.replace("[", "").replace("]", "").split("), (")
    polygon = ast.literal_eval(contents)

    n_bins = 200

    # Build the matplotlib figure
    # fig = plt.figure(figsize=(14, 5))
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5), sharex=True)

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    # y_bins = np.linspace(np.min(y_grid), np.max(y_grid), n_bins)

    y_curve = curve_df["vels"]

    # get freqs for dispersion curve
    # get freqs_grid with the same frequencies as the dispersion curve.
    curve_freqs = curve_df["freqs"]

    k_min, k_max = get_k_limits(site)

    residuals_freq, residuals_grid, quant_5, quant_95 = subset_data(
        subset_type,
        freqs_grid,
        vels_grid,
        curve_freqs,
        y_curve,
        polygon=None,
        k_limits=[k_min, k_max],
        y_max=y_max,
        remove_artifact=remove_artifact,
    )

    res_bins = np.linspace(np.min(residuals_grid), np.max(residuals_grid), n_bins)
    ax.hist2d(
        residuals_freq,
        residuals_grid,
        bins=[
            freq_bins,
            res_bins,
        ],
        norm=LogNorm(),
    )

    # ax.scatter(curve_freqs, quant_5, c="red")
    # ax.scatter(curve_freqs, quant_95, c="red")

    # smooth out data quantiles with rolling average
    """
    smoothed_quant_5 = (
        [np.mean(quant_5[:3])]
        + [np.mean(quant_5[i : i + 2]) for i in range(len(quant_5) - 2)]
        + [np.mean(quant_5[-3:])]
    )
    smoothed_quant_95 = (
        [np.mean(quant_95[:3])]
        + [np.mean(quant_95[i : i + 2]) for i in range(len(quant_95) - 2)]
        + [np.mean(quant_95[-3:])]
    )
    ax[0].plot(curve_freqs, smoothed_quant_5, c="red")
    ax[0].plot(curve_freqs, smoothed_quant_95, c="red")
    """
    # plot quantiles from error distribution fit
    # df = pd.read_csv("./figures/curve_fitting/WH01/asym-laplace-all/WH01-True.csv")
    # ax[0].plot(df["freqs"], df["q1"], c="orange", linestyle="--")
    # ax[0].plot(df["freqs"], df["q2"], c="orange", linestyle="--")

    # ax[0].axvline(x=3.0)

    # fit residuals

    x_data = curve_freqs  # [inds]

    quant_5 = np.array(quant_5)  # [inds]
    quant_95 = np.array(quant_95)  # [inds]

    ax.set_ylabel("residuals")

    # plt.colorbar(label="counts")

    # plt.show()

    return ax


def plot_spread(site, subset_type, y_max, remove_artifact, fig=None, ax=None):
    max_path, curve_path, polygon_path = get_path(site)

    df_max = read_max_file(max_path)
    freqs = np.unique(df_max["frequency"])
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]

    curve_df = pd.read_csv(curve_path)

    # if callback_context is None:
    with open(polygon_path) as f:
        contents = f.read()
    # polygon = contents.replace("[", "").replace("]", "").split("), (")
    polygon = ast.literal_eval(contents)

    # Build the matplotlib figure
    # fig = plt.figure(figsize=(14, 5))
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5), sharex=True)

    y_curve = curve_df["vels"]

    # get freqs for dispersion curve
    # get freqs_grid with the same frequencies as the dispersion curve.
    curve_freqs = curve_df["freqs"]

    k_min, k_max = get_k_limits(site)

    residuals_freq, residuals_grid, quant_5, quant_95 = subset_data(
        subset_type,
        freqs_grid,
        vels_grid,
        curve_freqs,
        y_curve,
        polygon=None,
        k_limits=[k_min, k_max],
        y_max=y_max,
        remove_artifact=remove_artifact,
    )

    # fit residuals

    x_data = curve_freqs  # [inds]

    quant_5 = np.array(quant_5)
    quant_95 = np.array(quant_95)

    # get None inds
    # inds = [
    #     ind
    #     for ind in range(len(x_data))
    #     if quant_5[ind] is None or quant_95[ind] is None
    # ]

    inds = []
    if site == "WH01":
        inds = [11, 56 + 1]
    elif site == "WH02":
        inds = [10, 72 + 1]
    elif site == "WH04":
        inds = [7, -1]
    if len(inds) > 0:
        x_spread = x_data[inds[0] : inds[1]]
        y_spread = quant_95[inds[0] : inds[1]] - quant_5[inds[0] : inds[1]]
        # y_ratio = np.abs(quant_95[max_ind:] / quant_5[max_ind:])
        # y_ratio = np.abs(quant_5[inds[0] : inds[1]] / quant_95[inds[0] : inds[1]])
        y_ratio = np.abs(quant_95[inds[0] : inds[1]] / quant_5[inds[0] : inds[1]])
        # y_ratio = quant_95[max_ind:] + quant_5[max_ind:]
    else:
        x_spread = x_data
        y_spread = quant_95 - quant_5
        y_ratio = np.abs(quant_95 / quant_5)
        # y_ratio = np.abs(quant_5 / quant_95)
        # y_ratio = quant_95 + quant_5

    # plt.colorbar(label="counts")

    # smooth out using a 3-point rolling average
    smoothed_spread = (
        [np.mean(y_spread[:3])]
        + [np.mean(y_spread[i : i + 2]) for i in range(len(y_spread) - 2)]
        + [np.mean(y_spread[-3:])]
    )
    smoothed_ratio = (
        [np.mean(y_ratio[:3])]
        + [np.mean(y_ratio[i : i + 2]) for i in range(len(y_ratio) - 2)]
        + [np.mean(y_ratio[-3:])]
    )

    # save smoothed spread to file
    df = pd.DataFrame(
        {
            "freq": x_spread,
            "spread": smoothed_spread,
            "ratio": smoothed_ratio,
        }
    )
    df.to_csv(
        "./results/curves/spread/"
        + site
        + "_smoothed_spread_"
        + str(remove_artifact)
        + ".csv"
    )

    # add end points

    ax.scatter(x_spread, y_spread, c="c")
    ax.plot(x_spread, smoothed_spread, c="red")

    ax2 = ax.twinx()
    ax2.scatter(x_spread, y_ratio, c="b")
    ax2.plot(x_spread, smoothed_ratio, c="red")

    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("spread")
    ax2.set_ylabel("ratio")

    # plt.show()
    return ax


def plot_residuals_parameterize(site, plot_polygon):
    """
    plot the histogram of f-k beamforming results with the selected dispersion curve subtracted.
    read in the polygon that was selected with the polygon picker.
    plot the 5th and 95th quantiles of the data. Plot the same quantiles for the error distribution fit.
    """

    # Exponential function model
    def test_exp(x, a, b):
        return a * np.exp(b * x)

    # with and without polygon slicing

    max_path, curve_path, polygon_path = get_path(site)

    df_max = read_max_file(max_path)
    # freqs_grid = np.unique(df_max["frequency"])
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]

    curve_df = pd.read_csv(curve_path)

    # if callback_context is None:
    with open(polygon_path) as f:
        contents = f.read()
    # polygon = contents.replace("[", "").replace("]", "").split("), (")
    polygon = ast.literal_eval(contents)

    n_bins = 200

    # Build the matplotlib figure
    # fig = plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(14, 5), sharex=True)

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    # y_bins = np.linspace(np.min(y_grid), np.max(y_grid), n_bins)

    y_curve = curve_df["vels"]

    # get freqs for dispersion curve
    # get freqs_grid with the same frequencies as the dispersion curve.
    curve_freqs = curve_df["freqs"]

    residuals_freq = []
    residuals_grid = []
    quant_5 = []
    quant_95 = []
    for f in curve_freqs:
        vels = vels_grid[np.isclose(freqs_grid, f)].values
        if plot_polygon:
            inds = [shapely.within(Point(f, v), Polygon(polygon)) for v in vels]
            res = list(vels[inds] - y_curve[curve_freqs == f].values[0])
        else:
            res = list(vels - y_curve[curve_freqs == f].values[0])
        residuals_freq += list(np.repeat(f, len(res)))
        residuals_grid += res
        quant_5.append(np.quantile(res, 0.05))
        quant_95.append(np.quantile(res, 0.95))

    res_bins = np.linspace(np.min(residuals_grid), np.max(residuals_grid), n_bins)
    ax[0].hist2d(
        residuals_freq,
        residuals_grid,
        bins=[
            freq_bins,
            res_bins,
        ],
        norm=LogNorm(),
    )

    df = pd.read_csv("./figures/curve_fitting/WH01/asym-laplace-all/WH01-True.csv")

    # ax[0].scatter(curve_freqs, quant_5, c="red")
    # ax[0].scatter(curve_freqs, quant_95, c="red")

    # smooth out data quantiles with rolling average
    smoothed_quant_5 = (
        [np.mean(quant_5[:3])]
        + [np.mean(quant_5[i : i + 2]) for i in range(len(quant_5) - 2)]
        + [np.mean(quant_5[-3:])]
    )
    smoothed_quant_95 = (
        [np.mean(quant_95[:3])]
        + [np.mean(quant_95[i : i + 2]) for i in range(len(quant_95) - 2)]
        + [np.mean(quant_95[-3:])]
    )
    ax[0].plot(curve_freqs, smoothed_quant_5, c="red")
    ax[0].plot(curve_freqs, smoothed_quant_95, c="red")

    # fit exponential to quantiles
    param_5, _ = curve_fit(test_exp, curve_freqs, smoothed_quant_5)
    ans_5 = param_5[0] * np.exp(param_5[1] * curve_freqs)
    ax[0].plot(curve_freqs, ans_5, c="red", linestyle="--")

    param_95, _ = curve_fit(test_exp, curve_freqs, smoothed_quant_95)
    ans_95 = param_95[0] * np.exp(param_95[1] * curve_freqs)
    ax[0].plot(curve_freqs, ans_95, c="red", linestyle="--")

    # plot quantiles from error distribution fit
    # ax[0].plot(df["freqs"], df["q1"], c="orange", linestyle="--")
    # ax[0].plot(df["freqs"], df["q2"], c="orange", linestyle="--")

    ax[0].set_title(str(param_5) + ", " + str(param_95))

    # ax[0].axvline(x=3.0)

    # fit residuals

    x_data = curve_freqs  # [inds]

    quant_5 = np.array(quant_5)  # [inds]
    quant_95 = np.array(quant_95)  # [inds]

    ax[0].set_ylabel("residuals")

    # plt.colorbar(label="counts")

    spread = quant_95 - quant_5

    # smooth out using a 3-point rolling average
    smoothed_spread = (
        [np.mean(spread[:3])]
        + [np.mean(spread[i : i + 2]) for i in range(len(spread) - 2)]
        + [np.mean(spread[-3:])]
    )
    # add end points

    ax[1].scatter(x_data, spread)
    ax[1].plot(x_data, smoothed_spread)

    # Fit exponential
    param, param_cov = curve_fit(test_exp, x_data, spread)
    ans = param[0] * np.exp(param[1] * x_data)

    ax[1].plot(x_data, ans, c="red")

    ax[1].set_xscale("log")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("spread")

    ax[1].set_title(param)

    plt.show()


def plot_dc_subplots(site, subset_type, y_max, remove_artifact):
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8, 15), sharex=True)

    plot_computed_dispersion_curve(
        site, y_max=y_max, plot_k_limits=True, fig=fig, ax=ax[0]
    )
    plot_residuals(
        site,
        subset_type,
        y_max=y_max,
        remove_artifact=remove_artifact,
        fig=fig,
        ax=ax[1],
    )
    plot_scaling_parameters(
        site, remove_artifact, remove_outliers=True, fig=fig, ax=ax[2:]
    )

    plt.suptitle(site)

    # plt.show()
    plt.savefig(
        "./figures/curve_fitting/scaling_params/full_"
        + site
        + "_"
        + str(remove_artifact)
        + "_inv"
        + ".png"
    )
