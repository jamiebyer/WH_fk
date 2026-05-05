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

from scipy.optimize import curve_fit


def asymmetric_laplacian(x, lambd, kappa, scale):
    mu = 0
    lambd = scale * lambd
    s = np.sign(x - mu)
    data_pred = (lambd / (kappa + 1 / kappa)) * np.exp(-(x - mu) * lambd * s * kappa**s)

    return data_pred


def def_asymmetric_laplacian(scale):
    def func(x, lambd, kappa):
        mu = 0
        lambd = scale * lambd
        s = np.sign(x - mu)
        data_pred = (lambd / (kappa + 1 / kappa)) * np.exp(
            -(x - mu) * lambd * s * kappa**s
        )

        return data_pred

    return func


def optimization_fitting_all(site, n_bins, polygon):
    # get scaling parameters from fitting data spread
    all_min, all_max, data_dict, scale = get_all_data(site, n_bins, polygon=polygon)

    freqs = data_dict.keys()
    q1_list, q2_list = [], []
    x = np.linspace(all_min, all_max, 100000)
    for ind, f in enumerate(freqs):
        param, param_cov = curve_fit(
            def_asymmetric_laplacian(scale[ind]),
            data_dict[f]["x"],
            data_dict[f]["counts"],
            method="dogbox",
        )
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)
        # plt.clf()

        res = data_dict[f]["res"]

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )

        # xbins = list(np.arange(min_res, 0, x_spacing)) + list(
        #     np.arange(0, max_res, x_spacing)
        # )
        # print(xbins, "\n")
        # xbins = np.linspace(min_res, max_res, n_bins)
        axs[0].hist(res, bins=xbins, density=True)
        # axs[0].set_ylim([0, 0.02])
        # axs[0].set_xlim([min_res, max_res])
        axs[0].set_xlim([all_min, all_max])

        lambd, kappa = param
        data_pred = asymmetric_laplacian(data_dict[f]["x"], lambd, kappa, scale[ind])
        pdf = asymmetric_laplacian(x, lambd, kappa, scale[ind])

        # plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)
        plot_distribution(axs, x, pdf)

        # save the quantiles for the saved distributions
        # integrate distribution
        dx = x[1] - x[0]
        cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
        # get quantiles and peak
        q1 = x[np.argmin(np.abs(cdf - 0.05))]
        q2 = x[np.argmin(np.abs(cdf - 0.95))]

        q1_list.append(q1)
        q2_list.append(q2)

        axs[1].axvline(x=q1, c="red")
        axs[1].axvline(x=q2, c="red")

        inds = data_dict[f]["counts"] != 0
        residuals = data_pred - data_dict[f]["counts"]
        axs[2].axhline(y=0, c="black")
        axs[2].scatter(data_dict[f]["x"][inds], residuals[inds])
        axs[2].set_ylim([-0.025, 0.025])

        # plt.suptitle(str(dist_params))
        plt.suptitle(
            "freq: " + str(np.round(f, 2)) + "\n"
            "lambda: "
            + str(np.round(lambd, 4))
            + ", "
            + "kappa: "
            + str(np.round(kappa, 4))
            + ", "
            + "scale: "
            + str(scale[ind])
        )

        plt.savefig(
            "./figures/curve_fitting/"
            + site
            + "/asym-laplace-all/"
            + site
            + "-"
            + str(np.round(f, 2))
            + "-"
            + str(polygon)
            + ".png"
        )
        plt.close()

    df = pd.DataFrame({"freqs": freqs, "q1": q1_list, "q2": q2_list})
    df.to_csv(
        "./figures/curve_fitting/"
        + site
        + "/asym-laplace-all/"
        + site
        + "-"
        + str(polygon)
        + ".csv"
    )


def all_data_distribution_fitting(site, dist, n_bins, freq_range, polygon):
    # get scaling parameters from fitting data spread
    all_min, all_max, curve_freqs, data_x, counts, all_res, data_dict = get_all_data(
        site, n_bins, freq_range, polygon=polygon
    )

    best_params = perform_grid_search(dist, all_min, all_max, data_x, counts)
    print("lambda: " + str(best_params[0]) + "kappa: " + str(best_params[1]))

    q1_list, q2_list = [], []
    x = np.linspace(all_min, all_max, 100000)
    for ind, f in enumerate(curve_freqs):
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)
        # plt.clf()

        res = data_dict[f]["res"]

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )

        axs[0].hist(res, bins=xbins, density=True)
        # axs[0].set_ylim([0, 0.02])
        # axs[0].set_xlim([min_res, max_res])
        axs[0].set_xlim([all_min, all_max])

        data_pred = get_distribution(dist, best_params, data_dict[f]["x"])
        pdf = get_distribution(dist, best_params, x)

        # plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)
        plot_distribution(axs, x, pdf)

        # save the quantiles for the saved distributions
        # integrate distribution
        dx = x[1] - x[0]
        cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
        # get quantiles and peak
        q1 = x[np.argmin(np.abs(cdf - 0.05))]
        q2 = x[np.argmin(np.abs(cdf - 0.95))]

        q1_list.append(q1)
        q2_list.append(q2)

        axs[1].axvline(x=q1, c="red")
        axs[1].axvline(x=q2, c="red")

        inds = data_dict[f]["counts"] != 0
        residuals = data_pred - data_dict[f]["counts"]
        axs[2].axhline(y=0, c="black")
        axs[2].scatter(data_dict[f]["x"][inds], residuals[inds])
        axs[2].set_ylim([-0.025, 0.025])

        # plt.suptitle(str(dist_params))
        plt.suptitle(
            "freq: "
            + str(np.round(f, 2))
            + "\n"
            + "lambda: "
            + str(np.round(best_params[0], 4))
            + ", kappa: "
            + str(np.round(best_params[1], 4))
        )

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

    """
    df = pd.DataFrame({"freqs": freqs, "q1": q1_list, "q2": q2_list})
    df.to_csv(
        "./figures/curve_fitting/"
        + site
        + "/"
        + dist
        + "-all"
        + "/"
        + site
        + "-"
        + str(polygon)
        + ".csv"
    )
    """
    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)

    x_spacing = (all_max - all_min) / n_bins
    xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
        np.arange(x_spacing / 2, max_res, x_spacing)
    )

    # plot all residuals together
    axs[0].hist(all_res, bins=xbins, density=True)
    # axs[0].set_ylim([0, 0.02])
    # axs[0].set_xlim([min_res, max_res])
    axs[0].set_xlim([all_min, all_max])

    data_pred = get_distribution(dist, best_params, data_dict[f]["x"])
    pdf = get_distribution(dist, best_params, x)

    # plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)
    plot_distribution(axs, x, pdf)

    # save the quantiles for the saved distributions
    # integrate distribution
    dx = x[1] - x[0]
    cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
    # get quantiles and peak
    q1 = x[np.argmin(np.abs(cdf - 0.05))]
    q2 = x[np.argmin(np.abs(cdf - 0.95))]

    q1_list.append(q1)
    q2_list.append(q2)

    axs[1].axvline(x=q1, c="red")
    axs[1].axvline(x=q2, c="red")

    inds = data_dict[f]["counts"] != 0
    residuals = data_pred - data_dict[f]["counts"]
    axs[2].axhline(y=0, c="black")
    axs[2].scatter(data_dict[f]["x"][inds], residuals[inds])
    axs[2].set_ylim([-0.025, 0.025])

    # plt.suptitle(str(dist_params))
    plt.suptitle(
        "lambda: "
        + str(np.round(best_params[0], 4))
        + ", kappa: "
        + str(np.round(best_params[1], 4))
    )

    plt.savefig(
        "./figures/curve_fitting/"
        + site
        + "/"
        + dist
        + "-all"
        + "/"
        + site
        + "-"
        + str(polygon)
        + ".png"
    )


def all_data_distribution_fitting_scale(site, dist, n_bins, polygon):
    # get scaling parameters from fitting data spread
    all_min, all_max, data_dict, scale = get_all_data_scale(
        site, n_bins, polygon=polygon
    )

    best_params = all_data_grid_search(dist, data_dict, scale)
    print("lambda: " + str(best_params[0]) + "kappa: " + str(best_params[1]))

    freqs = data_dict.keys()
    q1_list, q2_list = [], []
    x = np.linspace(all_min, all_max, 100000)
    for ind, f in enumerate(freqs):
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)
        # plt.clf()

        res = data_dict[f]["res"]

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )

        # xbins = list(np.arange(min_res, 0, x_spacing)) + list(
        #     np.arange(0, max_res, x_spacing)
        # )
        # print(xbins, "\n")
        # xbins = np.linspace(min_res, max_res, n_bins)
        axs[0].hist(res, bins=xbins, density=True)
        # axs[0].set_ylim([0, 0.02])
        # axs[0].set_xlim([min_res, max_res])
        axs[0].set_xlim([all_min, all_max])

        data_pred = get_distribution(dist, best_params, data_dict[f]["x"], scale[ind])
        pdf = get_distribution(dist, best_params, x, scale[ind])

        # plot_distribution(axs, x, pdf, cdf, dist_q1, dist_q2, dist_peak)
        plot_distribution(axs, x, pdf)

        # save the quantiles for the saved distributions
        # integrate distribution
        dx = x[1] - x[0]
        cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
        # get quantiles and peak
        q1 = x[np.argmin(np.abs(cdf - 0.05))]
        q2 = x[np.argmin(np.abs(cdf - 0.95))]

        q1_list.append(q1)
        q2_list.append(q2)

        axs[1].axvline(x=q1, c="red")
        axs[1].axvline(x=q2, c="red")

        inds = data_dict[f]["counts"] != 0
        residuals = data_pred - data_dict[f]["counts"]
        axs[2].axhline(y=0, c="black")
        axs[2].scatter(data_dict[f]["x"][inds], residuals[inds])
        axs[2].set_ylim([-0.025, 0.025])

        # plt.suptitle(str(dist_params))
        plt.suptitle(
            "freq: "
            + str(np.round(f, 2))
            + "\n"
            + "lambda: "
            + str(np.round(best_params[0], 4))
            + ", kappa: "
            + str(np.round(best_params[1], 4))
            + ", scale:"
            + str(scale[ind])
        )

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

    df = pd.DataFrame({"freqs": freqs, "q1": q1_list, "q2": q2_list, "scale": scale})
    df.to_csv(
        "./figures/curve_fitting/"
        + site
        + "/"
        + dist
        + "-all"
        + "/"
        + site
        + "-"
        + str(polygon)
        + ".csv"
    )


def distribution_fitting(site, selected_freq, n_bins, polygon):

    fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)  # , figsize=(3.5, 2.5)

    min_res, max_res, data_q1, data_q2, data_peak, data_x, data_y = get_data(
        axs, site, selected_freq, n_bins, polygon=polygon
    )

    # loop over a grid of possible params
    # dist = "normal"
    # dist = "EMG"
    # dist = "log-normal"
    dist = "asym-laplace"

    best_params = perform_grid_search(dist, min_res, max_res, data_x, data_y)
    lambd, kappa = best_params

    data_pred = get_distribution(dist, best_params, data_x)

    # x, pdf, cdf, dist_q1, dist_q2, dist_peak, data_pred = get_distribution(
    #     dist, dist_params, min_res, max_res, data_x
    # )

    x = np.linspace(min_res, max_res, 100000)
    pdf = get_distribution(dist, best_params, x)
    plot_distribution(axs, x, pdf)

    # print(len(data_pred))
    # print(len(data_y))
    residuals = data_pred - data_y
    # print(len(residuals), "\n")

    axs[0].scatter(data_x, data_y, s=3, c="black")

    axs[1].scatter(data_x, data_pred, s=3, c="black")

    axs[2].axhline(y=0, c="black")
    axs[2].scatter(data_x, residuals)
    # axs[2].set_ylim([-0.0025, 0.0025])

    # plt.suptitle(str(dist_params))
    plt.suptitle(
        "freq: " + str(np.round(f, 2)) + "\n"
        "lambda: "
        + str(np.round(lambd, 4))
        + ", "
        + "kappa: "
        + str(np.round(kappa, 4))
    )

    plt.savefig(
        "./figures/curve_fitting/"
        + site
        + "/asym-laplace/"
        + site
        + "-"
        + str(np.round(selected_freq, 2))
        + "-"
        + str(polygon)
        + ".png"
    )

    return lambd, kappa


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
        mu = 0
        # asymmetric laplacian params
        # mu_min, mu_max = -150, 150
        lambd_min, lambd_max = -3, 3
        kappa_min, kappa_max = -3, 1

        # mu = np.linspace(mu_min, mu_max, 100)
        lambd = np.logspace(lambd_min, lambd_max, 60)
        kappa = np.logspace(kappa_min, kappa_max, 60)
        params = list(itertools.product(lambd, kappa))

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
        # mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        lambd_min, lambd_max = best_params[0] * 0.95, best_params[0] * 1.05
        kappa_min, kappa_max = best_params[1] * 0.95, best_params[1] * 1.05

        # mu_fine = np.linspace(mu_min, mu_max, 60)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        kappa_fine = np.linspace(kappa_min, kappa_max, 60)

        params = list(itertools.product(lambd_fine, kappa_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = get_best_params(params, dist, min_res, max_res, data_x, data_y)

    return best_params


def all_data_grid_search(dist, data_dict, scale):
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

        # mu = np.linspace(mu_min, mu_max, 100)
        # sigma = np.linspace(sigma_min, sigma_max, 75)
        # lambd = np.logspace(lambd_min, lambd_max, 30)

        mu = np.linspace(mu_min, mu_max, 15)
        sigma = np.linspace(sigma_min, sigma_max, 15)
        lambd = np.logspace(lambd_min, lambd_max, 15)
        params = list(itertools.product(mu, sigma, lambd))
    elif dist == "asym-laplace":
        # asymmetric laplacian params
        # mu_min, mu_max = -150, 150
        lambd_min, lambd_max = -3, 3
        kappa_min, kappa_max = -3, 1

        # mu = np.linspace(mu_min, mu_max, 100)
        lambd = np.logspace(lambd_min, lambd_max, 60)
        kappa = np.logspace(kappa_min, kappa_max, 60)
        # params = list(itertools.product(mu, lambd, kappa))
        params = list(itertools.product(lambd, kappa))

    best_params = all_get_best_params(params, dist, data_dict, scale)

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

        # mu_fine = np.linspace(mu_min, mu_max, 60)
        # sigma_fine = np.linspace(sigma_min, sigma_max, 60)
        # lambd_fine = np.linspace(lambd_min, lambd_max, 40)

        mu_fine = np.linspace(mu_min, mu_max, 15)
        sigma_fine = np.linspace(sigma_min, sigma_max, 15)
        lambd_fine = np.linspace(lambd_min, lambd_max, 15)
        params = list(itertools.product(mu_fine, sigma_fine, lambd_fine))
    elif dist == "asym-laplace":
        # mu_min, mu_max = best_params[0] * 0.95, best_params[0] * 1.05
        lambd_min, lambd_max = best_params[0] * 0.95, best_params[0] * 1.05
        kappa_min, kappa_max = best_params[1] * 0.95, best_params[1] * 1.05

        # mu_fine = np.linspace(mu_min, mu_max, 60)
        lambd_fine = np.linspace(lambd_min, lambd_max, 40)
        kappa_fine = np.linspace(kappa_min, kappa_max, 60)

        params = list(itertools.product(lambd_fine, kappa_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = all_get_best_params(params, dist, data_dict, scale)

    return best_params


def get_best_params(params, dist, min_res, max_res, data_x, data_y):
    best_params = None
    best_lsq = np.inf
    for p in params:
        inds = data_y != 0
        data_pred = get_distribution(dist, p, data_x)
        # discretize distribution on same frequencies as data

        # compare with data...
        residuals = data_pred[inds] - data_y[inds]
        lsq = np.sum(residuals**2)
        # logL = -np.sum((residuals**2) / (2 * sigma_data**2))
        if lsq < best_lsq:
            best_lsq = lsq
            best_params = p

    return best_params


def all_get_best_params(params, dist, data_dict, scale):
    best_params = None
    best_lsq = np.inf
    for p in params:
        lsq = 0
        for i, (f, val) in enumerate(data_dict.items()):
            inds = val["counts"] != 0
            data_pred = get_distribution(dist, p, val["x"][inds], scale[i])
            # remove data with 0 counts

            # compare with data...
            residuals = data_pred - val["counts"][inds]
            lsq += (1 / np.sum(inds)) * (np.sum(residuals**2))
            # logL = -np.sum((residuals**2) / (2 * sigma_data**2))

        if lsq < best_lsq:
            best_lsq = lsq
            best_params = p

    return best_params


def get_distribution(dist, dist_params, data_x, scale=1):
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
        mu = 0
        lambd, kappa = dist_params
        # s = np.sign(x - mu)
        # pdf = (lambd / (kappa + 1 / kappa)) * np.exp(-(x - mu) * lambd * s * kappa**s)
        lambd = scale * lambd
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

    peaks, _ = find_peaks(pdf)
    peak = None
    if len(peaks) == 1:
        peak = x[peaks[0]]
    """
    # return x, pdf, cdf, q1, q2, peak, data_pred
    return data_pred


def plot_distribution(axs, x, pdf):
    axs[0].plot(x, pdf)

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


def get_all_data(site, n_bins, freq_range, polygon=False):
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

    inds = (curve_freqs >= freq_range[0]) & (curve_freqs <= freq_range[1])
    curve_freqs = curve_freqs[inds]
    y_curve = y_curve[inds]

    # polygon info
    if polygon:
        with open(polygon_path) as f:
            contents = f.read()
        polygon = ast.literal_eval(contents)

    data_dict = {}
    # save points that are within the polygon
    all_res = []
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

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )
        # print(xbins, "\n")
        # xbins = np.linspace(min_res, max_res, n_bins)
        counts, bins, _ = plt.hist(res, bins=xbins, density=True)

        q_5 = np.quantile(res, 0.05)
        q_95 = np.quantile(res, 0.95)

        # values from the distribution will be the counts of the histogram at the midpoint of the bins...
        data_x = (bins[:-1] + bins[1:]) / 2
        data_dict[f] = {
            "x": data_x,
            "counts": counts,
            "res": res,
            "quant_5": q_5,
            "quant_95": q_95,
        }

        all_res += res

    min_res = np.min(all_res)
    max_res = np.max(all_res)

    # make sure xbins are centered on 0
    if min_res > 0 or max_res < 0:
        raise ValueError

    x_spacing = (max_res - min_res) / n_bins
    # xbins = list(np.arange(min_res, 0, x_spacing)) + list(
    #     np.arange(0, max_res, x_spacing)
    # )
    xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
        np.arange(x_spacing / 2, max_res, x_spacing)
    )
    # print(xbins, "\n")
    # xbins = np.linspace(min_res, max_res, n_bins)
    counts, bins, _ = plt.hist(all_res, bins=xbins, density=True)

    # q_5 = np.quantile(all_res, 0.05)
    # q_95 = np.quantile(all_res, 0.95)
    # ind = np.argmax(counts)
    # peak = (bins[ind] + bins[ind + 1]) / 2

    # values from the distribution will be the counts of the histogram at the midpoint of the bins...
    data_x = (bins[:-1] + bins[1:]) / 2

    return min_res, max_res, curve_freqs, data_x, counts, all_res, data_dict


def get_all_data_scale(site, n_bins, polygon=False):
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
    spread = []
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
        # xbins = list(np.arange(min_res, 0, x_spacing)) + list(
        #     np.arange(0, max_res, x_spacing)
        # )
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )
        # print(xbins, "\n")
        # xbins = np.linspace(min_res, max_res, n_bins)
        counts, bins, _ = plt.hist(res, bins=xbins, density=True)

        q_5 = np.quantile(res, 0.05)
        q_95 = np.quantile(res, 0.95)
        # ind = np.argmax(counts)
        # peak = (bins[ind] + bins[ind + 1]) / 2

        spread.append(q_95 - q_5)

        # values from the distribution will be the counts of the histogram at the midpoint of the bins...
        data_x = (bins[:-1] + bins[1:]) / 2
        data_dict[f] = {
            "x": data_x,
            "counts": counts,
            "res": res,
            "quant_5": q_5,
            "quant_95": q_95,
        }

    scale = 1 / np.array(spread)
    # fit exponential to spread of the data
    # param, param_cov = curve_fit(test_exp, curve_freqs, spread)
    # scale = param[0] * np.exp(param[1] * curve_freqs)
    # normalize scale
    # scale = (scale - scale.min()) / (scale.max() - scale.min())
    # scale = 1 - scale

    return all_min, all_max, data_dict, scale


def get_path(site, transverse_comp=False):
    max_path, curve_path = "", ""

    if site == "WH01":
        if transverse_comp:
            max_path = "./results/fk/final/conventionaltransverse-WH01-default04.max"
            curve_path = "./results/curves/og/curve-WH01-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH01_3C_split-default08.max"
            curve_path = "./results/curves/curve-WH01-vertical-velocity-False.csv"
    elif site == "WH02":
        if transverse_comp:
            max_path = "./results/fk/final/conventionaltransverse-WH02-default04.max"
            curve_path = "./results/curves/og/curve-WH02-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH02_3C_split-default08.max"
            curve_path = "./results/curves/curve-WH02-vertical-velocity-False.csv"
    elif site == "WH03":
        if transverse_comp:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH03-sliced-default04.max"
            )
            curve_path = "./results/curves/og/curve-WH03-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH03-default08.max"
            curve_path = "./results/curves/og/curve-WH03-1C.csv"
    elif site == "WH04":
        if transverse_comp:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH04-longest-default04.max"
            )
            curve_path = "./results/curves/og/curve-WH04-2C.csv"
        else:
            max_path = "./results/fk/final/conventional-WH04-longest-default08.max"
            curve_path = "./results/curves/curve-WH04-vertical-velocity-False.csv"

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
    """
    plot the histogram of f-k beamforming results with the selected dispersion curve subtracted.
    read in the polygon that was selected with the polygon picker.
    plot the 5th and 95th quantiles of the data. Plot the same quantiles for the error distribution fit.
    """
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


# Exponential function model
def test_exp(x, a, b):
    return a * np.exp(b * x)


def pick_curves():
    # possible curves

    # params that give test asymmetric laplacian noise
    # normal curve to use to invert it

    sigma_data = noise_percent * data_true
    data_obs = data_true + sigma_data * np.random.randn(len(periods))

    lambd, kappa = 0.086, 0.92
    # lambd, kappa = 0.086, 0.72

    x = np.linspace(-100, 100, 100000)
    mu = 0
    lambd, kappa = noise_params
    # lambd = scale * lambd

    s = np.sign(x - mu)
    pdf = (lambd / (kappa + 1 / kappa)) * np.exp(-(x - mu) * lambd * s * kappa**s)

    # integrate distribution
    # the cdf should go from 0 to 1
    dx = x[1] - x[0]
    cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)

    noise = []
    for _ in range(len(data_true)):
        # generate a random uniform number between 0 and 1
        n = np.random.uniform(0, 1)

        # use to select value from inverse of cdf
        ind = np.argmin(np.abs(cdf - n))
        x_pick = (x[ind] + x[ind + 1]) / 2

        noise.append(x_pick)

    data_obs = data_true + noise


if __name__ == "__main__":
    site = "WH04"

    # get list of possible frequencies to compute for selected site
    max_path, curve_path, polygon_path = get_path(site)
    # df_max = read_max_file(max_path)
    # freqs = np.unique(df_max["frequency"])
    curve_df = pd.read_csv(curve_path)
    freqs = curve_df["freqs"]

    # distribution_fitting(site=site, selected_freq=None, n_bins=60, polygon=False)

    # distribution_fitting(site=site, selected_freq=3.4901715488706766, n_bins=50)
    # distribution_fitting(
    #     site=site, selected_freq=7.254554357068926, n_bins=60, polygon=True
    # )

    plot_residuals(site, plot_polygon=True)

    """
    # for frequencies less than 3 Hz, find one set of params
    all_data_distribution_fitting(
        site, dist="asym-laplace", n_bins=60, freq_range=[0, 3], polygon=True
    )

    # get params individually for higher frequencies
    lambd_list, kappa_list = [], []
    out_freqs = []
    for f in freqs[freqs > 3]:
        lambd, kappa = distribution_fitting(site, f, n_bins=60, polygon=True)
        out_freqs.append(f)
        lambd_list.append(lambd)
        kappa_list.append(kappa)

    df = pd.DataFrame({"freqs": out_freqs, "lambd": lambd_list, "kappa": kappa_list})
    df.to_csv("./figures/curve_fitting/WH01/asym-laplace/params.csv")
    """

    # all_data_distribution_fitting_scale(
    #     site, dist="asym-laplace", n_bins=60, polygon=True
    # )
    # lambda: 0.11352397886304381, kappa: 0.8359713871778693

    # optimization_fitting_all(site, n_bins=60, polygon=True)
