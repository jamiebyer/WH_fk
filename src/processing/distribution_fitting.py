import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import itertools

from scipy import special
from scipy.signal import find_peaks


def distribution_fitting(site, selected_freq, n_bins):

    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)

    min_res, max_res, data_q1, data_q2, data_peak, data_x, data_y = get_data(
        axs, site, selected_freq, n_bins, polygon=False
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
        "./curve_fitting/"
        + site
        + "/"
        + site
        + "-"
        + str(np.round(selected_freq, 2))
        + ".png"
    )


def perform_grid_search(dist, min_res, max_res, data_x, data_y):
    """
    Search with coarse grid, then search with fine grid.
    """

    # make coarse grid of params to search for first iteration.
    if dist == "normal":
        # Gaussian params
        mu = np.linspace(-50, 50, 200)
        sigma = np.linspace(0.001, 200, 200)
        params = list(itertools.product(mu, sigma))
    elif dist == "log-normal":
        # log-normal params
        mu = np.linspace(-50, 50, 100)
        sigma = np.logspace(-3, 3, 100)
        params = list(itertools.product(mu, sigma))
    elif dist == "EMG":
        # EMG params
        mu_min, mu_max = -100, 100
        sigma_min, sigma_max = 0.001, 150
        lambd_min, lambd_max = -3, 2

        mu = np.linspace(mu_min, mu_max, 75)
        sigma = np.linspace(sigma_min, sigma_max, 75)
        lambd = np.logspace(lambd_min, lambd_max, 30)
        params = list(itertools.product(mu, sigma, lambd))

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)

    # define range for fine grid based on values for params
    if dist == "EMG":
        mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        sigma_min, sigma_max = best_params[1] * 0.95, best_params[1] * 1.05
        lambd_min, lambd_max = best_params[2] * 0.95, best_params[2] * 1.05

        mu_fine = np.linspace(mu_min, mu_max, 60)
        sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        # lambd = np.logspace(-3, 2, 30)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        params = list(itertools.product(mu_fine, sigma_fine, lambd_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)

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


def get_distribution(dist, dist_params, min_res, max_res, data_x):
    x = np.linspace(min_res, max_res, 100000)

    if dist == "EMG":
        mu, sigma, lambd = dist_params
        pdf = (
            (lambd / 2)
            * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * x))
            * (1 - special.erf((mu + lambd * sigma**2 - x) / (np.sqrt(2) * sigma)))
        )
        data_pred = (
            (lambd / 2)
            * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * data_x))
            * (1 - special.erf((mu + lambd * sigma**2 - data_x) / (np.sqrt(2) * sigma)))
        )
    elif dist == "normal":
        mu, sigma = dist_params
        pdf = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((x - mu) ** 2 / (2 * sigma**2))
        )
        data_pred = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((data_x - mu) ** 2 / (2 * sigma**2))
        )
    elif dist == "log-normal":
        mu, sigma = dist_params
        pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(x) - mu) ** 2) / (2 * sigma**2)
        )
        data_pred = (1 / (data_x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(data_x) - mu) ** 2) / (2 * sigma**2)
        )
        data_pred[np.isnan(data_pred)] = 0

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

    return x, pdf, cdf, q1, q2, peak, data_pred


def plot_distribution(axs, x, pdf, cdf, q1, q2, peak):
    axs[0].plot(x, pdf)

    axs[1].plot(x, pdf)
    axs[1].axvline(x=q1, c="black", alpha=0.5)
    axs[1].axvline(x=q2, c="black", alpha=0.5)
    if peak is not None:
        axs[1].axvline(x=peak, c="red", alpha=0.5)

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

    # RESIDUALS

    res = list(
        vels_grid[np.isclose(freqs_grid, selected_freq)].values
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


if __name__ == "__main__":
    site = "WH01"
    # get list of possible frequencies to compute for selected site
    max_path, curve_path, polygon_path = get_path(site)
    df_max = read_max_file(max_path)
    freqs = np.unique(df_max["frequency"])

    distribution_fitting(site=site, selected_freq=3.4901715488706766, n_bins=50)
