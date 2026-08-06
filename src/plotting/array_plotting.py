import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

from obspy.signal.array_analysis import array_transff_wavenumber

# ARRAY LAYOUT


def plot_array_layout(site):
    # plot array layout from relative positions
    # label instruments and indicate the odd ones
    data_path = "./data/" + site + "/" + site + "_loc_corrected_geopsy.txt"
    # data_path = "./data/" + site + "/txt_files/" + site + "_loc_corrected_geopsy.txt"

    df = pd.read_csv(data_path, names=["instrument", "x", "y"], sep="\s+")

    for i, row in df.iterrows():
        instrument = row["instrument"].replace("SS_", "")
        # if site == "WH03" and ((instrument == "25242") or (instrument == "25057")):
        #    color = "red"
        # elif site == "WH04" and ((instrument == "24625") or (instrument == "25257")):
        #     color = "red"
        # else:
        color = "black"

        plt.scatter(row["x"], row["y"], c=color)

    # plt.title(site)
    plt.xlabel("x (m)", fontsize=20)
    plt.ylabel("y (m)", fontsize=20)

    plt.tick_params(axis="both", which="major", labelsize=16)

    # plt.xlim([-75, 140])
    # plt.ylim([-75, 140])
    plt.show()


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


def plot_array_response(site):
    # https://docs.obspy.org/tutorial/code_snippets/array_response_function.html
    # https://geophydog.cool/post/array_response_function/#__31-the-geometry-effects__

    if site == "WH01" or site == "WH02":
        data_path = (
            "./data/" + site + "/txt_files/" + site + "_loc_corrected_geopsy.txt"
        )
    elif site == "WH03" or site == "WH04":
        data_path = data_path = (
            "./data/" + site + "/" + site + "_loc_corrected_geopsy.txt"
        )

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

    # convert coordinates from m to km
    coords /= 1000.0

    dists = []
    for c1 in coords:
        for c2 in coords:
            if np.sum(np.abs(c1) - np.abs(c2)) == 0.0:
                continue
            d = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            dists.append(d)

    # minumum distance between stations
    d_min = np.min(dists)
    # maximum distance between stations
    d_max = np.max(dists)

    # kmax (in rad/km)
    k_max = 2 * np.pi / d_min
    k_min = 2 * np.pi / d_max

    # set limits for wavenumber differences to analyze
    # klim = 500.0
    # klim = 7000.0
    klim = 8000.0
    # klim = 7500.0
    # klim = 1500.0
    # klim = 250.0
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

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # plot
    ax1.scatter(coords_df["x"], coords_df["y"])
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2.pcolor(kx / 1000, ky / 1000, transff.T)

    # plt.xlim(kxmin / 1000, kxmax / 1000)
    # plt.ylim(kymin / 1000, kymax / 1000)

    ax2.add_patch(plt.Circle((0, 0), k_min / 1000, color="r", fill=False))
    ax2.add_patch(plt.Circle((0, 0), k_max / 1000, color="r", fill=False))

    ax2.set_xlabel("k_x (rad/m)")
    ax2.set_ylabel("k_y (rad/m)")
    # plt.colorbar(label="array response")

    # xg, yg = np.meshgrid(kx / 1000, ky / 1000)
    interp = RegularGridInterpolator((kx / 1000, ky / 1000), transff.T)

    # for each angle, create a line and use interpolation to find the k_x and k_y vals for that line
    for t in np.linspace(0, 360, 1000):
        # make a line from 0 to 0.
        # r = np.linspace(0, 0.5, 250)
        # r = np.linspace(0, klim / 1000, 250)
        r = np.linspace(0, klim / 1000, 10000)
        x = r * np.cos(np.deg2rad(t))
        y = r * np.sin(np.deg2rad(t))
        # Xi, Yi = np.meshgrid(xi, yi)
        mag = np.sqrt(x**2 + y**2)

        z = interp((x, y))

        ax3.plot(mag, z, c="grey", alpha=0.20)
        # ax3.plot(mag, z, c="black")

    ax3.axvline(x=k_min / 1000)
    ax3.axvline(x=k_max / 1000)

    ax3.axhline(y=0.5)

    ax3.set_xlabel("k magnitude")
    ax3.set_ylabel("array response")

    plt.suptitle(
        site
        + "\nk (rad/m): "
        + str(k_min / 1000)
        + ", "
        + str(k_max / 1000)
        + "\nd (m): "
        + str(d_min * 1000)
        + ", "
        + str(d_max * 1000)
    )
    plt.tight_layout()
    plt.show()


def plot_array_response_individual(site, d_range=[]):
    # https://docs.obspy.org/tutorial/code_snippets/array_response_function.html
    # https://geophydog.cool/post/array_response_function/#__31-the-geometry-effects__

    if site == "WH01" or site == "WH02":
        data_path = (
            "./data/" + site + "/txt_files/" + site + "_loc_corrected_geopsy.txt"
        )
    elif site == "WH03" or site == "WH04":
        data_path = data_path = (
            "./data/" + site + "/" + site + "_loc_corrected_geopsy.txt"
        )

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

    # convert coordinates from m to km
    coords /= 1000.0

    dists = []
    for c1 in coords:
        for c2 in coords:
            if np.sum(np.abs(c1) - np.abs(c2)) == 0.0:
                continue
            d = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            dists.append(d)

    # minumum distance between stations
    d_min = np.min(dists)
    # maximum distance between stations
    d_max = np.max(dists)

    # kmax (in rad/km)
    k_max = 2 * np.pi / d_min
    k_min = 2 * np.pi / d_max

    # get dists to center
    diams = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    coords = coords[(diams == 0) | ((diams >= d_range[0]) & (diams <= d_range[1])), :]

    # set limits for wavenumber differences to analyze
    # klim = 500.0
    # klim = 1250.0
    # klim = 1500.0
    klim = 7000.0
    # klim = 7500.0
    # klim = 250.0
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

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # plot
    ax1.scatter(coords_df["x"], coords_df["y"], c="blue", s=4)
    ax1.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000, c="black", s=7)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2.pcolor(kx / 1000, ky / 1000, transff.T)

    # plt.xlim(kxmin / 1000, kxmax / 1000)
    # plt.ylim(kymin / 1000, kymax / 1000)

    ax2.add_patch(plt.Circle((0, 0), k_min / 1000, color="r", fill=False))
    ax2.add_patch(plt.Circle((0, 0), k_max / 1000, color="r", fill=False))

    ax2.set_xlabel("k_x (rad/m)")
    ax2.set_ylabel("k_y (rad/m)")
    # plt.colorbar(label="array response")

    # xg, yg = np.meshgrid(kx / 1000, ky / 1000)
    interp = RegularGridInterpolator((kx / 1000, ky / 1000), transff.T)

    # for each angle, create a line and use interpolation to find the k_x and k_y vals for that line
    for t in np.linspace(0, 360, 1000):
        # make a line from 0 to 0.
        # r = np.linspace(0, 0.5, 250)
        r = np.linspace(0, klim / 1000, 250)
        x = r * np.cos(np.deg2rad(t))
        y = r * np.sin(np.deg2rad(t))
        # Xi, Yi = np.meshgrid(xi, yi)
        mag = np.sqrt(x**2 + y**2)

        z = interp((x, y))

        ax3.plot(mag, z, c="grey", alpha=0.08)
        # ax3.plot(mag, z, c="black")

    ax3.axvline(x=k_min / 1000)
    ax3.axvline(x=k_max / 1000)

    ax3.axhline(y=0.5)

    ax3.set_xlabel("k magnitude")
    ax3.set_ylabel("array response")

    plt.suptitle(
        site
        + "\nk (rad/m): "
        + str(k_min / 1000)
        + ", "
        + str(k_max / 1000)
        + "\nd (m): "
        + str(d_min * 1000)
        + ", "
        + str(d_max * 1000)
    )
    plt.tight_layout()
    plt.show()
