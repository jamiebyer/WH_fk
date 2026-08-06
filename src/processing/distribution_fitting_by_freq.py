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

from utils.utils import read_max_file, get_path, get_k_limits


def get_data(
    site, n_bins, subset_type, polygon, k_limits, y_max, remove_artifact, x_scale
):
    """
    :param site:
    :param n_bins:
    :param polygon:

    clip data to subset within polygon.
    get histogram of residuals to fit the error distribution to.
    calculate quantiles.
    """
    max_path, curve_path, polygon_path = get_path(site)

    # max file / 2d hist df
    df_max = read_max_file(max_path)
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]

    # dispersion curve df
    curve_df = pd.read_csv(curve_path)
    in_curve_freqs = curve_df["freqs"].values
    in_y_curve = curve_df["vels"].values

    # subset to same dimensions as scale...
    curve_freqs, y_curve = [], []
    for i in range(len(in_curve_freqs)):
        if np.isclose(in_curve_freqs[i], x_scale).any():
            curve_freqs.append(in_curve_freqs[i])
            y_curve.append(in_y_curve[i])

    curve_freqs = np.array(curve_freqs)
    y_curve = np.array(y_curve)

    # read in polygon
    if polygon:
        with open(polygon_path) as f:
            contents = f.read()
        polygon = ast.literal_eval(contents)

    data_dict = {}
    # save points that are within the polygon
    all_res = []
    for ind, f in enumerate(curve_freqs):
        vels = vels_grid[np.isclose(freqs_grid, f)].values

        if subset_type == "polygon":
            # subset with polygon
            inds = [shapely.within(Point(f, v), Polygon(polygon)) for v in vels]
            res = list(vels[inds] - y_curve[curve_freqs == f].values[0])
        elif subset_type == "k_limits":
            # subset with the k limits
            # k = 2*pi*f / v_p
            # v_1 = 2*pi*f/k
            max_1 = np.array(2 * np.pi * f / k_limits[0])  # smaller max
            min_1 = np.array(2 * np.pi * f / k_limits[1])  # smaller min
            max_2 = np.array(2 * np.pi * f / (k_limits[0] / 2))  # larger max
            min_2 = np.array(2 * np.pi * f / (k_limits[1] / 2))  # larger min

            # original subset (with ymax)
            # res = list(
            #     vels[(vels >= min_1) & (vels <= max_2) & (vels <= y_max)]
            #     - y_curve[curve_freqs == f].values[0]
            # )
            # use k limits to select dispersion curve vals (but not spread of hist)

            if (y_curve[curve_freqs == f][0] >= min_2) and (
                y_curve[curve_freqs == f][0] <= max_2
            ):
                if remove_artifact:
                    res = list(
                        vels[(vels >= min_2) & (vels <= y_max)]
                        - y_curve[curve_freqs == f][0]
                    )
                else:
                    res = list(vels[vels <= y_max] - y_curve[curve_freqs == f][0])
            else:
                res = []

        else:
            res = list(vels - y_curve[curve_freqs == f].values[0])

        if not res:
            continue

        res = list(np.array(res) / 1000)

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )
        counts, bins, _ = plt.hist(res, bins=xbins, density=True)

        q_5 = np.quantile(res, 0.05)
        q_95 = np.quantile(res, 0.95)

        # values from the distribution will be the counts of the histogram at the midpoint of the bins...
        data_x = (bins[:-1] + bins[1:]) / 2
        data_dict[f] = {
            "xbins": xbins,
            "x_data": data_x,
            "counts": counts,
            "res": res,
            "quant_5": q_5,
            "quant_95": q_95,
        }

        all_res += res

    # the min and max of residuals from all frequencies
    min_res = np.min(all_res)
    max_res = np.max(all_res)

    # make sure xbins are centered on 0
    x_spacing = (max_res - min_res) / n_bins
    xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
        np.arange(x_spacing / 2, max_res, x_spacing)
    )
    counts, bins, _ = plt.hist(all_res, bins=xbins, density=True)

    # values from the distribution will be the counts of the histogram at the midpoint of the bins...
    data_x = (bins[:-1] + bins[1:]) / 2

    return min_res, max_res, curve_freqs, data_x, counts, all_res, data_dict


def asymmetric_laplacian(x, lambd, kappa, scale, mu=0):
    """
    :param x:
    :param lambd: lambda, scale parameter. Inversely correlated to spread
    :param kappa: skewness parameter
    :param scale:

    Asymmetric Laplacian distribution.

    """
    lambd = scale * lambd
    s = np.sign(x - mu)
    data_pred = (lambd / (kappa + 1 / kappa)) * np.exp(-(x - mu) * lambd * s * kappa**s)

    return data_pred


def get_distribution(dist, dist_params, data_x, lambd_scale=1, kappa_scale=1):
    """
    :param dist: distribution type ("EMG", "normal", "log-normal", "asym-laplace")
    :param dist_params:
    :param data_x:
    :param scale:
    """
    if dist == "EMG":
        mu, sigma, lambd = dist_params
        data_pred = (
            (lambd / 2)
            * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * data_x))
            * (1 - special.erf((mu + lambd * sigma**2 - data_x) / (np.sqrt(2) * sigma)))
        )
    elif dist == "normal":
        mu, sigma = dist_params
        data_pred = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
            -((data_x - mu) ** 2 / (2 * sigma**2))
        )
    elif dist == "log-normal":
        mu, sigma = dist_params
        data_pred = (1 / (data_x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log(data_x) - mu) ** 2) / (2 * sigma**2)
        )
        data_pred[np.isnan(data_pred)] = 0
    elif dist == "asym-laplace":
        mu = 0
        lambd, kappa = dist_params
        lambd = lambd_scale * lambd
        kappa = kappa_scale * kappa
        s = np.sign(data_x - mu)
        data_pred = (lambd / (kappa + 1 / kappa)) * np.exp(
            -(data_x - mu) * lambd * s * kappa**s
        )

    return data_pred


def run_grid_search(dist, data_dict, selected_freq):
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
        kappa_min, kappa_max = -3, 2

        # mu = np.linspace(mu_min, mu_max, 100)
        lambd = np.logspace(lambd_min, lambd_max, 60)
        kappa = np.logspace(kappa_min, kappa_max, 60)
        # params = list(itertools.product(mu, lambd, kappa))
        params = list(itertools.product(lambd, kappa))

    best_params = get_best_params(params, dist, data_dict[selected_freq])

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
        lambd_fine = np.linspace(lambd_min, lambd_max, 60)
        kappa_fine = np.linspace(kappa_min, kappa_max, 60)

        params = list(itertools.product(lambd_fine, kappa_fine))

    # if the chosen params are one of the bounds, need to expand the range.

    best_params = get_best_params(params, dist, data_dict[selected_freq])

    return best_params


def get_best_params(params, dist, data_dict):
    """ """
    best_params = None
    best_lsq = np.inf
    for p in params:
        # remove data with 0 counts
        inds = data_dict["counts"] != 0

        data_pred = get_distribution(dist, p, data_dict["x_data"][inds])

        nan_inds = np.isnan(data_pred)

        # compare with data...
        residuals = data_pred[~nan_inds] - data_dict["counts"][inds][~nan_inds]

        # only use non-nan, and non-zero count indices...
        term = (1 / sum(~nan_inds)) * (np.sum(residuals**2))
        lsq = term
        # logL = -np.sum((residuals**2) / (2 * sigma_data**2))

        if lsq < best_lsq:
            best_lsq = lsq
            best_params = p

        # print(lsq, p)
        # print(best_lsq, best_params, "\n")

    return best_params


def optimization_fitting(site, n_bins, polygon):
    """
    Test doing fitting using scipy.optimize curve_fit.
    """

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

    # get scaling parameters from fitting data spread
    all_min, all_max, data_dict, scale = get_data(site, n_bins, polygon=polygon)

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
        axs[0].plot(x, pdf)
        axs[1].plot(x, pdf)

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


def error_distribution_fitting_by_individual_frequencies(
    site, dist, n_bins, subset_type
):
    """
    :param site: Site name ("WH01", "WH02", "WH03", "WH04")
    :param dist: Distribution type ("normal", "asym-laplace")
    :param n_bins:
    """

    # df = pd.read_csv("./results/curves/spread/" + site + "_smoothed_spread_True.csv")
    df = pd.read_csv("./results/curves/spread/" + site + "_smoothed_spread_False.csv")
    x_scale = df["freq"]

    # read in data and get histograms...
    k_min, k_max = get_k_limits(site)
    all_min, all_max, curve_freqs, data_x, counts, all_res, data_dict = get_data(
        site,
        n_bins,
        subset_type,
        polygon=None,
        k_limits=[k_min, k_max],
        y_max=1200,
        # remove_artifact=True,
        remove_artifact=False,
        x_scale=x_scale,
    )

    # lists for storing quantiles and params at each frequency
    q1_list, q2_list = [], []
    params_list = []
    x = np.linspace(all_min, all_max, 100000)  # x for plotting pdf
    # loop over each frequency in dispersion curve frequencies
    for ind, f in enumerate(data_dict.keys()):
        best_params = run_grid_search(dist, data_dict, f)

        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)

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

        data_pred = get_distribution(
            dist,
            best_params,
            data_dict[f]["x_data"],
        )
        pdf = get_distribution(
            dist,
            best_params,
            x,
        )

        axs[0].plot(x, pdf)
        axs[1].plot(x, pdf)

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
        axs[2].scatter(data_dict[f]["x_data"][inds], residuals[inds])
        # axs[2].set_ylim([-0.25, 0.25])

        params_list.append(best_params)

        plt.savefig(
            "./figures/curve_fitting/"
            + site
            + "-"
            + dist
            + "/"
            + site
            + "-"
            + str(np.round(f, 2))
            + ".png"
        )
        plt.close()

    params_list = np.array(params_list)

    df_dict = {"freqs": curve_freqs, "q1": q1_list, "q2": q2_list}
    if dist == "normal":
        pass
    elif dist == "asym-laplace":
        df_dict["lambd"] = params_list[:, 0]
        df_dict["kappa"] = params_list[:, 1]

    df = pd.DataFrame(df_dict)
    df.to_csv("./results/curve_fitting/" + site + "-" + dist + "-params.csv")


def get_noise_pdf(noise_dist, noise_params, freq_ind=None, mu=0):
    x = np.linspace(-50, 50, 100000)

    if noise_dist == "normal":
        std_data = noise_params["std"]

        if isinstance(std_data, float):
            std = std_data
        else:
            std = std_data[freq_ind]

        pdf = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(
            -((x - mu) ** 2 / (2 * std**2))
        )

    if noise_dist == "asym-laplace":
        lambd, kappa = noise_params["lambd"], noise_params["kappa"]
        lambd_scaling = noise_params["lambd_scale"]

        lambd = (1 / lambd_scaling) * lambd

        if isinstance(lambd, float):
            l = lambd
        else:
            l = lambd[freq_ind]

        s = np.sign(x - mu)
        pdf = (l / (kappa + 1 / kappa)) * np.exp(-(x - mu) * l * s * kappa**s)

    # integrate distribution
    # the cdf should go from 0 to 1
    dx = x[1] - x[0]
    cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)

    # q_low, q_high = 0.159, 0.841
    q_5 = x[np.argmin(np.abs(cdf - 0.05))]
    q_95 = x[np.argmin(np.abs(cdf - 0.95))]

    return x, pdf, cdf, q_5, q_95


def generate_noise_dist(freqs, noise_dist, noise_params):
    # can give the noise frequency-dependent scaling using either
    # a percent of the true data
    # or an exponential based on values from fitting the spread/percentiles of the field data

    # lower: 15.9, higher: 84.1, to have 68.2 range
    AL_q_lower_list, AL_q_higher_list = [], []
    norm_q_lower_list, norm_q_higher_list = [], []

    freqs_2d, noise_2d = [], []
    stds = []
    for ind in range(len(freqs)):
        x, pdf, cdf, q_5, q_95 = get_noise_pdf(noise_dist, noise_params, freq_ind=ind)
        AL_q_lower_list.append(q_5)
        AL_q_higher_list.append(q_95)

        picks = []
        for _ in range(10000):
            # generate a random uniform number between 0 and 1
            n = np.random.uniform(0, 1)

            # use to select value from inverse of cdf
            i = np.argmin(np.abs(cdf - n))
            x_pick = (x[i] + x[i + 1]) / 2

            picks.append(x_pick)

        std = np.std(picks)
        stds.append(std)

        x, pdf, cdf, q_5, q_95 = get_noise_pdf(
            noise_dist="normal", noise_params={"std": std}, freq_ind=ind
        )
        norm_q_lower_list.append(q_5)
        norm_q_higher_list.append(q_95)

        freqs_2d += len(picks) * [freqs[ind]]
        noise_2d += picks

    stds = np.array(stds)

    return (
        freqs_2d,
        noise_2d,
        AL_q_lower_list,
        AL_q_higher_list,
        norm_q_lower_list,
        norm_q_higher_list,
        stds,
    )


def get_simulated_data(n_bins):
    """ """
    # generate simulated data
    noise_dist = "normal"
    # noise_dist = "asym-laplace"
    noise_params = {"frequency_scaling": False, "std": 0.075}
    """
    noise_params = {
        "frequency_scaling": False,
        "lambd_scale": 1,
        "lambd": 6.8,
        "kappa": 0.72,
    }
    """
    n_data = 100
    freqs = 1 / np.logspace(0, 1.1, n_data)
    (
        freqs_grid,
        vels_grid,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    ) = generate_noise_dist(freqs, noise_dist, noise_params)

    freqs_grid = np.array(freqs_grid)
    vels_grid = np.array(vels_grid)
    curve_freqs = np.unique(freqs_grid)

    data_dict = {}
    # save points that are within the polygon
    all_res = []
    for ind, f in enumerate(curve_freqs):
        res = vels_grid[np.isclose(freqs_grid, f)]

        min_res = np.min(res)
        max_res = np.max(res)

        x_spacing = (max_res - min_res) / n_bins
        xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
            np.arange(x_spacing / 2, max_res, x_spacing)
        )
        counts, bins, _ = plt.hist(res, bins=xbins, density=True)

        q_5 = np.quantile(res, 0.05)
        q_95 = np.quantile(res, 0.95)

        # values from the distribution will be the counts of the histogram at the midpoint of the bins...
        data_x = (bins[:-1] + bins[1:]) / 2
        data_dict[f] = {
            "xbins": xbins,
            "x_data": data_x,
            "counts": counts,
            "res": res,
            "quant_5": q_5,
            "quant_95": q_95,
        }

        all_res += list(res)

    # the min and max of residuals from all frequencies
    min_res = np.min(all_res)
    max_res = np.max(all_res)

    # make sure xbins are centered on 0
    x_spacing = (max_res - min_res) / n_bins
    xbins = list(np.flip(np.arange(-x_spacing / 2, min_res, -x_spacing))) + list(
        np.arange(x_spacing / 2, max_res, x_spacing)
    )
    counts, bins, _ = plt.hist(all_res, bins=xbins, density=True)

    # values from the distribution will be the counts of the histogram at the midpoint of the bins...
    data_x = (bins[:-1] + bins[1:]) / 2

    return min_res, max_res, curve_freqs, data_x, counts, all_res, data_dict


def fit_simulated_dataset(n_bins=60):
    dist = "asym-laplace"

    # read in data and get histograms...
    all_min, all_max, curve_freqs, data_x, counts, all_res, data_dict = (
        get_simulated_data(n_bins)
    )

    # grid search to get best distribution parameters
    best_params = run_grid_search(dist, data_dict)

    # lists for storing quantiles and params at each frequency
    q1_list, q2_list = [], []
    params_list = []
    x = np.linspace(all_min, all_max, 100000)  # x for plotting pdf

    # loop over each frequency in dispersion curve frequencies
    for ind, f in enumerate(curve_freqs):
        fig, axs = plt.subplots(ncols=1, nrows=3, sharex=True)

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
        # axs[0].set_xlim([all_min, all_max])

        data_pred = get_distribution(dist, best_params, data_dict[f]["x_data"])
        pdf = get_distribution(dist, best_params, x)

        axs[0].plot(x, pdf)
        axs[1].plot(x, pdf)

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
        axs[2].scatter(data_dict[f]["x_data"][inds], residuals[inds])
        # axs[2].set_ylim([-0.025, 0.025])

        params_list.append(best_params)

        plt.savefig(
            "./figures/curve_fitting/"
            + "sim-data"
            + "/"
            + dist
            + "/"
            + "sim-data"
            + "-"
            + str(np.round(f, 2))
            + ".png"
        )
        plt.close()

    params_list = np.array(params_list)

    df_dict = {"freqs": curve_freqs, "q1": q1_list, "q2": q2_list}
    if dist == "normal":
        pass
    elif dist == "asym-laplace":
        df_dict["lambd"] = params_list[:, 0]
        df_dict["kappa"] = params_list[:, 1]

    df = pd.DataFrame(df_dict)
    df.to_csv("./results/curve_fitting/" + "sim-data" + "-" + dist + "-params.csv")
