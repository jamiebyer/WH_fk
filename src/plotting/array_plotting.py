import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ARRAY RESPONSE


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
        data_path = ""

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
    plt.subplot(1, 3, 1)
    # plt.subplot(3, 1, 1)
    plt.scatter(coords_df["x"], coords_df["y"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(1, 3, 2)
    # plt.subplot(3, 1, 2)
    plt.pcolor(kx / 1000, ky / 1000, transff.T)

    # plt.xlim(kxmin / 1000, kxmax / 1000)
    # plt.ylim(kymin / 1000, kymax / 1000)

    plt.Circle((0, 0), k_max / 1000, color="r")

    plt.xlabel("k_x (rad/m)")
    plt.ylabel("k_y (rad/m)")
    plt.colorbar(label="array response")

    plt.subplot(1, 3, 3)
    # plt.subplot(3, 1, 3)
    for yind in range(len(ky)):
        k_mag = np.sqrt(kx**2 * ky[yind] ** 2)
        inds = np.argsort(k_mag)
        # plt.scatter(
        #    k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005, s=3
        # )
        plt.plot(k_mag[inds] / 1000000, transff[inds, yind], c="grey", alpha=0.005)

    # plt.axvline(x=k_min / 1000)
    # plt.axvline(x=k_max / 1000)

    plt.axhline(y=0.5)

    plt.xlabel("k magnitude")
    plt.ylabel("array response")

    plt.suptitle("WH01 array transfer function")
    # plt.tight_layout()
    plt.show()
