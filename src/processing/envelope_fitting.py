import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy import stats

from utils.utils import read_max_file, get_path, get_k_limits


def get_data(site, y_max, remove_artifact):
    """
    :param site:
    :param n_bins:
    :param polygon:

    clip data to subset within polygon.
    get histogram of residuals to fit the error distribution to.
    calculate quantiles.
    """
    k_limits = get_k_limits(site)
    max_path, curve_path, _ = get_path(site)

    # max file / 2d hist df
    df_max = read_max_file(max_path)
    freqs_grid = df_max["frequency"]
    vels_grid = 1 / df_max["slowness"]

    # dispersion curve df
    curve_df = pd.read_csv(curve_path)
    curve_freqs = curve_df["freqs"].values
    y_curve = curve_df["vels"].values

    # save points that are within the polygon
    points = []
    weights = []
    for ind, f in enumerate(curve_freqs):
        vels = vels_grid[np.isclose(freqs_grid, f)].values

        # subset with the k limits
        # k = 2*pi*f / v_p
        # v_1 = 2*pi*f/k
        max_1 = np.array(2 * np.pi * f / k_limits[0])  # smaller max
        min_1 = np.array(2 * np.pi * f / k_limits[1])  # smaller min
        max_2 = np.array(2 * np.pi * f / (k_limits[0] / 2))  # larger max
        min_2 = np.array(2 * np.pi * f / (k_limits[1] / 2))  # larger min

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

        res = np.array(res) / 1000

        if len(res) > 0:
            points += [[f, r] for r in res]
            weights += [1 / len(res)] * len(res)

    return np.array(points), weights


def get_CI():
    # get 95% CI from PDF
    pass


def KDE_fitting(site, y_max, remove_artifact):
    points, weights = get_data(site, y_max, remove_artifact)

    kern = stats.gaussian_kde(points.T, bw_method=0.75, weights=weights)
    """
    kde = KernelDensity(
        kernel="gaussian",
        # bandwidth="scott",
        # bandwidth="silverman",
        bandwidth=0.3,
        # breadth_first=False,
    ).fit(points)
    """
    # make a grid of points in the sample space
    X, Y = np.meshgrid(
        np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100),
        np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100),
    )
    plot_points = np.array([X.flatten(), Y.flatten()]).T

    # prob = np.reshape(np.exp(kde.score_samples(plot_points)), (100, 100))
    prob = np.reshape(np.exp(kern(plot_points.T)), (100, 100))

    plt.hist2d(points[:, 0], points[:, 1], bins=[50, 50], cmin=1)
    plt.colorbar()
    # plt.scatter(points[:, 0], points[:, 1], c="black", s=3)
    plt.contour(X, Y, prob)

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Residuals (km/s)")
    plt.title("KDE, bw: 0.3")
    plt.show()


def make_ellipses(gmm, n_comp, ax):
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    # colors = ["navy", "turquoise", "darkorange"]
    for n in range(n_comp):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle  # , color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        # ax.set_aspect("equal", "datalim")


def GMM_fitting(site, y_max, remove_artifact):
    n_comp = 3

    points = get_data(site, y_max, remove_artifact)
    gmm = GaussianMixture(n_components=n_comp, covariance_type="full").fit(points)
    # density = gmm.predict([[0, 0], [12, 3]])

    means = gmm.means_
    covariances = gmm.covariances_

    # make a grid of points in the sample space
    X, Y = np.meshgrid(
        np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 500),
        np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 500),
    )
    plot_points = np.array([X.flatten(), Y.flatten()]).T

    prob = np.reshape(np.exp(gmm.score_samples(plot_points)), (500, 500))

    fig, ax = plt.subplots(ncols=1, nrows=1)

    # plt.hist2d(points[:, 0], points[:, 1], bins=[50, 50], cmin=1)
    plt.scatter(points[:, 0], points[:, 1], c="black", s=3)
    plt.contour(X, Y, prob)

    plt.scatter(means[:, 0], means[:, 1], c="red", zorder=1)

    # make_ellipses(gmm, n_comp, ax)

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Residuals (km/s)")
    plt.title(
        "n_comp: "
        + str(n_comp)
        + ", cov: full"
        # + ", remove artifact: True"
        + ", bic: "
        + str(np.round(gmm.bic(points), 2))
    )
    plt.show()
