import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import itertools

from scipy import special
from scipy.signal import find_peaks

import shapely
from shapely import LineString, Point, Polygon

from matplotlib.colors import LogNorm


def all_data_distribution_fitting(site, dist, n_bins, polygon):

    all_min, all_max, data_dict = get_all_data(site, n_bins, polygon=polygon)

    best_params = all_data_grid_search(dist, data_dict)

    x = np.linspace(all_min, all_max, 100000)
    for f in data_dict.keys():
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)
        # plt.clf()

        res = data_dict[f]["res"]

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.arange(min_res, 0, x_spacing)) + list(
            np.arange(0, max_res, x_spacing)
        )
        print(xbins, "\n")
        # xbins = np.linspace(min_res, max_res, n_bins)
        axs[0].hist(res, bins=xbins, density=True)
        # axs[0].set_ylim([0, 0.02])
        axs[0].set_xlim([min_res, max_res])

        data_pred = get_distribution(dist, best_params, data_dict[f]["x"])
        pdf = get_distribution(dist, best_params, x)

        # plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)
        plot_distribution(axs, x, pdf)

        inds = data_dict[f]["counts"] != 0
        residuals = data_pred - data_dict[f]["counts"]
        axs[2].axhline(y=0, c="black")
        axs[2].scatter(data_dict[f]["x"][inds], residuals[inds])
        axs[2].set_ylim([-0.0025, 0.0025])

        # plt.suptitle(str(dist_params))
        plt.suptitle(str(best_params))

        plt.savefig(
            "./figures/curve_fitting/"
            + site
            + "/"
            + dist
            + "-all"
            + "/"
            + site
            + "-"
            + str(np.round(f, 2))
            + "-"
            + str(polygon)
            + ".png"
        )
        plt.close()


def distribution_fitting(site, selected_freq, n_bins, polygon):

    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)

    min_res, max_res, data_q1, data_q2, data_peak, data_x, data_y = get_data(
        axs, site, selected_freq, n_bins, polygon=polygon
    )

    # loop over a grid of possible params
    # dist = "normal"
    dist = "EMG"
    # dist = "log-normal"
    # dist_params = [0, 10, 1]  # mu, sigma, lambd
    dist_params = [0, 0.25]  # mu, sigma
    # q1, q2, peak = get_distribution(axs, dist, dist_params, min_res, max_res)

    """
    if dist == "normal":
        # Gaussian params
        mu = np.linspace(30, 100, 200)
        sigma = np.linspace(100, 200, 200)
        params = list(itertools.product(mu, sigma))
    elif dist == "log-normal":
        # log-normal params
        mu = np.linspace(-50, 50, 100)
        sigma = np.logspace(-3, 3, 100)
        params = list(itertools.product(mu, sigma))
    elif dist == "EMG":
        # EMG params
        mu = np.linspace(-80, -20, 40)
        sigma = np.linspace(15, 75, 60)
        lambd = np.logspace(-3, 1, 20)
        params = list(itertools.product(mu, sigma, lambd))

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)
    """

    best_params = perform_grid_search(dist, min_res, max_res, data_x, data_y)

    x, pdf, cdf, dist_q1, dist_q2, dist_peak, data_pred = get_distribution(
        dist, best_params, min_res, max_res, data_x
    )

    # x, pdf, cdf, dist_q1, dist_q2, dist_peak, data_pred = get_distribution(
    #     dist, dist_params, min_res, max_res, data_x
    # )
    plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)

    residuals = data_pred - data_y
    axs[2].axhline(y=0, c="black")
    axs[2].scatter(data_x, residuals)
    # axs[2].set_ylim([-0.0025, 0.0025])

    # plt.suptitle(str(dist_params))
    plt.suptitle(str(best_params))

    plt.savefig(
        "./figures/curve_fitting/"
        + site
        + "/"
        + site
        + "-"
        + str(np.round(selected_freq, 2))
        + "-"
        + str(polygon)
        + ".png"
    )


def perform_grid_search(dist, min_res, max_res, data_x, data_y):
    """
    Search with coarse grid, then search with fine grid.
    """

    # make coarse grid of params to search for first iteration.
    if dist == "normal":
        # Gaussian params
        mu_min, mu_max = -150, 150
        sigma_min, sigma_max = 0.001, 150

        mu = np.linspace(mu_min, mu_max, 200)
        sigma = np.linspace(sigma_min, sigma_max, 200)
        params = list(itertools.product(mu, sigma))
    elif dist == "log-normal":
        # log-normal params
        mu = np.linspace(-50, 50, 100)
        sigma = np.logspace(-3, 3, 100)
        params = list(itertools.product(mu, sigma))
    elif dist == "EMG":
        # EMG params
        mu_min, mu_max = -150, 150
        sigma_min, sigma_max = 0.001, 150
        lambd_min, lambd_max = -3, 2

        mu = np.linspace(mu_min, mu_max, 100)
        sigma = np.linspace(sigma_min, sigma_max, 75)
        lambd = np.logspace(lambd_min, lambd_max, 30)
        params = list(itertools.product(mu, sigma, lambd))
    elif dist == "asym-laplace":
        # asymmetric laplacian params
        mu_min, mu_max = -150, 150
        lambd_min, lambd_max = -3, 3
        kappa_min, kappa_max = -3, 1

        mu = np.linspace(mu_min, mu_max, 100)
        lambd = np.logspace(lambd_min, lambd_max, 60)
        kappa = np.logspace(kappa_min, kappa_max, 60)
        params = list(itertools.product(mu, lambd, kappa))

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)

    # define range for fine grid based on values for params
    if dist == "normal":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        sigma_min, sigma_max = best_params[1] * 0.95, best_params[1] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        params = list(itertools.product(mu_fine, sigma_fine))
    elif dist == "EMG":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        sigma_min, sigma_max = best_params[1] * 0.95, best_params[1] * 1.05
        lambd_min, lambd_max = best_params[2] * 0.95, best_params[2] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        # lambd = np.logspace(-3, 2, 30)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        params = list(itertools.product(mu_fine, sigma_fine, lambd_fine))
    elif dist == "asym-laplace":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        lambd_min, lambd_max = best_params[1] * 0.95, best_params[1] * 1.05
        kappa_min, kappa_max = best_params[2] * 0.95, best_params[2] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        kappa_fine = np.linspace(kappa_min, kappa_max, 60)

        params = list(itertools.product(mu_fine, lambd_fine, kappa_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)

    return best_params


def all_data_grid_search(dist, data_dict):
    """
    Search with coarse grid, then search with fine grid.
    """

    # make coarse grid of params to search for first iteration.
    if dist == "normal":
        # Gaussian params
        mu_min, mu_max = -150, 150
        sigma_min, sigma_max = 0.001, 150

        mu = np.linspace(mu_min, mu_max, 200)
        sigma = np.linspace(sigma_min, sigma_max, 200)
        params = list(itertools.product(mu, sigma))
    elif dist == "log-normal":
        # log-normal params
        mu = np.linspace(-50, 50, 100)
        sigma = np.logspace(-3, 3, 100)
        params = list(itertools.product(mu, sigma))
    elif dist == "EMG":
        # EMG params
        mu_min, mu_max = -150, 150
        sigma_min, sigma_max = 0.001, 150
        lambd_min, lambd_max = -3, 2

        mu = np.linspace(mu_min, mu_max, 100)
        sigma = np.linspace(sigma_min, sigma_max, 75)
        lambd = np.logspace(lambd_min, lambd_max, 30)

        # mu = np.linspace(mu_min, mu_max, 6)
        # sigma = np.linspace(sigma_min, sigma_max, 6)
        # lambd = np.logspace(lambd_min, lambd_max, 6)
        params = list(itertools.product(mu, sigma, lambd))
    elif dist == "asym-laplace":
        # asymmetric laplacian params
        mu_min, mu_max = -150, 150
        lambd_min, lambd_max = -3, 3
        kappa_min, kappa_max = -3, 1

        mu = np.linspace(mu_min, mu_max, 100)
        lambd = np.logspace(lambd_min, lambd_max, 60)
        kappa = np.logspace(kappa_min, kappa_max, 60)
        params = list(itertools.product(mu, lambd, kappa))

    best_params = all_get_best_params(params, dist, data_dict)

    # define range for fine grid based on values for params
    if dist == "normal":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        sigma_min, sigma_max = best_params[1] * 0.95, best_params[1] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        params = list(itertools.product(mu_fine, sigma_fine))
    elif dist == "EMG":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        sigma_min, sigma_max = best_params[1] * 0.95, best_params[1] * 1.05
        lambd_min, lambd_max = best_params[2] * 0.95, best_params[2] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        # lambd = np.logspace(-3, 2, 30)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        params = list(itertools.product(mu_fine, sigma_fine, lambd_fine))
    elif dist == "asym-laplace":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        lambd_min, lambd_max = best_params[1] * 0.95, best_params[1] * 1.05
        kappa_min, kappa_max = best_params[2] * 0.95, best_params[2] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        kappa_fine = np.linspace(kappa_min, kappa_max, 60)

        params = list(itertools.product(mu_fine, lambd_fine, kappa_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = all_get_best_params(params, dist, data_dict)

    return best_params


def get_best_params(params, dist, min_res, max_res, data_x, data_y):
    best_params = None
    best_lsq = np.inf
    for p in params:
        x, pdf, cdf, dist_q1, dist_q2, dist_peak, data_pred = get_distribution(
            dist, p, min_res, max_res, data_x
        )
        # discretize distribution on same frequencies as data

        # compare with data...
        residuals = data_pred - data_y
        lsq = np.sum(residuals**2)
        # logL = -np.sum((residuals**2) / (2 * sigma_data**2))
        if lsq < best_lsq:
            best_lsq = lsq
            best_params = p

    return best_params


def all_get_best_params(params, dist, data_dict):
    best_params = None
    best_lsq = np.inf
    for p in params:
        lsq = 0
        for f, val in data_dict.items():
            inds = val["counts"] != 0
            data_pred = get_distribution(dist, p, val["x"][inds])
            # remove data with 0 counts

            # compare with data...
            residuals = data_pred - val["counts"][inds]
            lsq += (1 / np.sum(inds)) * (np.sum(residuals**2))
            # logL = -np.sum((residuals**2) / (2 * sigma_data**2))

        if lsq < best_lsq:
            best_lsq = lsq
            best_params = p

    return best_params


def get_distribution(dist, dist_params, data_x):
    # x = np.linspace(min_res, max_res, 100000)

    if dist == "EMG":
        mu, sigma, lambd = dist_params
        # pdf = (
        #     (lambd / 2)
        #     * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * x))
        #     * (1 - special.erf((mu + lambd * sigma**2 - x) / (np.sqrt(2) * sigma)))
        # )
        data_pred = (
            (lambd / 2)
            * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * data_x))
            * (1 - special.erf((mu + lambd * sigma**2 - data_x) / (np.sqrt(2) * sigma)))
        )
    elif dist == "normal":
        mu, sigma = dist_params
        # pdf = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
        #     -((x - mu) ** 2 / (2 * sigma**2))
        # )
        data_pred = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((data_x - mu) ** 2 / (2 * sigma**2))
        )
    elif dist == "log-normal":
        mu, sigma = dist_params
        # pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
        #     -((np.log(x) - mu) ** 2) / (2 * sigma**2)
        # )
        data_pred = (1 / (data_x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(data_x) - mu) ** 2) / (2 * sigma**2)
        )
        data_pred[np.isnan(data_pred)] = 0
    elif dist == "asym-laplace":
        mu, lambd, kappa = dist_params
        # s = np.sign(x - mu)
        # pdf = (lambd / (kappa + 1 / kappa)) * np.exp(-(x - mu) * lambd * s * kappa**s)
        s = np.sign(data_x - mu)
        data_pred = (lambd / (kappa + 1 / kappa)) * np.exp(
            -(data_x - mu) * lambd * s * kappa**s
        )
    """
    # integrate distribution
    dx = x[1] - x[0]
    cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
    # get quantiles and peak
    q1 = x[np.argmin(np.abs(cdf - 0.05))]
    q2 = x[np.argmin(np.abs(cdf - 0.95))]

    # axs[0].plot(x, pdf, c="orange")

    peaks, _ = find_peaks(pdf)
    peak = None
    if len(peaks) == 1:
        peak = x[peaks[0]]
    """
    # return x, pdf, cdf, q1, q2, peak, data_pred
    return data_pred


def plot_distribution(axs, x, pdf):
    # axs[0].plot(x, pdf)

    axs[1].plot(x, pdf)
    # axs[1].axvline(x=q1, c="black", alpha=0.5)
    # axs[1].axvline(x=q2, c="black", alpha=0.5)
    # if peak is not None:
    #     axs[1].axvline(x=peak, c="red", alpha=0.5)

    """
    axs[2].plot((x[:-1] + x[1:]) / 2, cdf)
    # plot peak and quantiles
    axs[2].axvline(x=q1, c="black", alpha=0.5)
    axs[2].axvline(x=q2, c="black", alpha=0.5)
    axs[2].axvline(x=peak, c="red", alpha=0.5)
    # axs[2].set_xlim([min_res, max_res])
    """


def get_data(axs, site, selected_freq, n_bins, polygon=False):
    # read data in, get distribution for specific frequency
    # (with or without polygon)
    # calculate residuals...
    # calculate quantiles and get peak

    max_path, curve_path, polygon_path = get_path(site)

    # max file / 2d hist df
    df_max = read_max_file(max_path)
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]
    # freqs_curve = np.unique(freqs_grid)

    # dispersion curve df
    curve_df = pd.read_csv(curve_path)
    curve_freqs = curve_df["freqs"]
    y_curve = curve_df["vels"]

    # polygon info
    if polygon:
        with open(polygon_path) as f:
            contents = f.read()
        polygon = ast.literal_eval(contents)

        # select data which is within the polygon
        vels = vels_grid[np.isclose(freqs_grid, selected_freq)].values
        inds = [shapely.within(Point(selected_freq, v), Polygon(polygon)) for v in vels]

    # RESIDUALS

    res = list(
        # vels_grid[np.isclose(freqs_grid, selected_freq)].values
        vels[inds]
        - y_curve[curve_freqs == selected_freq].values[0]
    )
    min_res = np.min(res)
    max_res = np.max(res)

    xbins = np.linspace(min_res, max_res, n_bins)
    counts, bins, _ = axs[0].hist(res, bins=xbins, density=True)
    q1 = np.quantile(res, 0.05)
    q2 = np.quantile(res, 0.95)
    ind = np.argmax(counts)
    peak = (bins[ind] + bins[ind + 1]) / 2

    axs[0].axvline(x=q1, c="black")
    axs[0].axvline(x=q2, c="black")
    if peak is not None:
        axs[0].axvline(x=peak, c="red")
    axs[0].set_xlim([min_res, max_res])

    # values from the distribution will be the counts of the histogram at the midpoint of the bins...
    data_x = (bins[:-1] + bins[1:]) / 2

    return min_res, max_res, q1, q2, peak, data_x, counts


def get_all_data(site, n_bins, polygon=False):
    # read data in, get distribution for specific frequency
    # (with or without polygon)
    # calculate residuals...
    # calculate quantiles and get peak

    max_path, curve_path, polygon_path = get_path(site)

    # max file / 2d hist df
    df_max = read_max_file(max_path)
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]
    # freqs_curve = np.unique(freqs_grid)

    # dispersion curve df
    curve_df = pd.read_csv(curve_path)
    curve_freqs = curve_df["freqs"]
    y_curve = curve_df["vels"]

    # polygon info
    if polygon:
        with open(polygon_path) as f:
            contents = f.read()
        polygon = ast.literal_eval(contents)

    data_dict = {}
    all_min, all_max = 0, 0
    for f in curve_freqs:
        vels = vels_grid[np.isclose(freqs_grid, f)].values
        if polygon:
            # select data which is within the polygon
            inds = [shapely.within(Point(f, v), Polygon(polygon)) for v in vels]

            res = list(
                # vels_grid[np.isclose(freqs_grid, selected_freq)].values
                vels[inds]
                - y_curve[curve_freqs == f].values[0]
            )
        else:
            res = list(
                # vels_grid[np.isclose(freqs_grid, selected_freq)].values
                vels
                - y_curve[curve_freqs == f].values[0]
            )

        min_res = np.min(res)
        max_res = np.max(res)

        # make sure xbins are centered on 0
        if min_res > 0 or max_res < 0:
            raise ValueError
        if min_res < all_min:
            all_min = min_res
        if max_res > all_max:
            all_max = max_res

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.arange(min_res, 0, x_spacing)) + list(
            np.arange(0, max_res, x_spacing)
        )
        # xbins = np.linspace(min_res, max_res, n_bins)
        counts, bins, _ = plt.hist(res, bins=xbins, density=True)

        # q1 = np.quantile(res, 0.05)
        # q2 = np.quantile(res, 0.95)
        # ind = np.argmax(counts)
        # peak = (bins[ind] + bins[ind + 1]) / 2

        # values from the distribution will be the counts of the histogram at the midpoint of the bins...
        data_x = (bins[:-1] + bins[1:]) / 2

        data_dict[f] = {"x": data_x, "counts": counts, "res": res}

    return all_min, all_max, data_dict


def get_path(site, transverse_comp=False):
    max_path, curve_path = "", ""

    if site == "WH01":
        if transverse_comp:
            max_path = "./results/fk/final/conventionaltransverse-WH01-default04.max"
            curve_path = "./results/curves/curve-WH01-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH01_3C_split-default08.max"
            curve_path = "./results/curves/curve-WH01-1C.csv"
    elif site == "WH02":
        if transverse_comp:
            max_path = "./results/fk/final/conventionaltransverse-WH02-default04.max"
            curve_path = "./results/curves/curve-WH02-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH02_3C_split-default08.max"
            curve_path = "./results/curves/curve-WH02-1C.csv"
    elif site == "WH03":
        if transverse_comp:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH03-sliced-default04.max"
            )
            curve_path = "./results/curves/curve-WH03-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH03-default08.max"
            curve_path = "./results/curves/curve-WH03-1C.csv"
    elif site == "WH04":
        if transverse_comp:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH04-longest-default04.max"
            )
            curve_path = "./results/curves/curve-WH04-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH04-longest-default08.max"
            curve_path = "./results/curves/curve-WH04-1C.csv"

    polygon_path = curve_path.replace(".csv", ".txt")

    return max_path, curve_path, polygon_path


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
    # df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)
    df = pd.read_csv(max_file, skiprows=ind, sep="\\s", names=names)
    return df


def plot_residuals(site, plot_polygon):
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
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 5))

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
    # quant_5 = []
    # quant_95 = []
    for f in curve_freqs:
        vels = vels_grid[np.isclose(freqs_grid, f)].values
        if plot_polygon:
            inds = [shapely.within(Point(f, v), Polygon(polygon)) for v in vels]
            res = list(vels[inds] - y_curve[curve_freqs == f].values[0])
        else:
            res = list(vels - y_curve[curve_freqs == f].values[0])
        residuals_freq += list(np.repeat(f, len(res)))
        residuals_grid += res
        # quant_5.append(np.quantile(res, 0.05))
        # quant_95.append(np.quantile(res, 0.95))

    res_bins = np.linspace(np.min(residuals_grid), np.max(residuals_grid), n_bins)
    plt.hist2d(
        residuals_freq,
        residuals_grid,
        bins=[
            freq_bins,
            res_bins,
        ],
        norm=LogNorm(),
    )

    plt.xscale("log")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("residuals")

    plt.colorbar(label="counts")

    plt.show()


if __name__ == "__main__":
    site = "WH01"
    # get list of possible frequencies to compute for selected site
    max_path, curve_path, polygon_path = get_path(site)
    df_max = read_max_file(max_path)
    freqs = np.unique(df_max["frequency"])

    # distribution_fitting(site=site, selected_freq=None, n_bins=60, polygon=False)

    # distribution_fitting(site=site, selected_freq=3.4901715488706766, n_bins=50)
    # distribution_fitting(
    #     site=site, selected_freq=7.254554357068926, n_bins=60, polygon=True
    # )

    # plot_residuals(site, plot_polygon=True)

    all_data_distribution_fitting(site, dist="EMG", n_bins=60, polygon=False)
