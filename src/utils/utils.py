import os
import datetime

import numpy as np
import pandas as pd

import shapely
from shapely import Point, Polygon


def make_output_folder(dir_path):
    if not os.path.isdir(dir_path) and not os.path.isfile(dir_path):
        os.mkdir(dir_path)


def create_file_list(ind, in_path):
    files = []
    for station in os.listdir(in_path):
        for file in os.listdir(in_path + station):
            files.append((station, file.split(".")[0]))

    return files[ind][0], files[ind][1]


# functions for parsing xml


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def is_date(val):
    try:
        datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False


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


def get_k_limits(site):
    """
    Get wavenumber limits from site array geometry.
    """

    if site in ["WH01", "WH02"]:
        data_path = (
            "./data/" + site + "/txt_files/" + site + "_loc_corrected_geopsy.txt"
        )
    elif site in ["WH03", "WH04"]:
        data_path = "./data/" + site + "/" + site + "_loc_corrected_geopsy.txt"

    df = pd.read_csv(data_path, names=["instrument", "x", "y"], sep="\s+")

    x = df["x"].values
    y = df["y"].values

    dist = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if (x[i] == x[j]) and (y[i] == y[j]):
                continue
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            dist.append(d)

    d_min = np.min(dist)
    d_max = np.max(dist)
    print(d_min, d_max)

    k_min = 2 * np.pi / d_max
    k_max = 2 * np.pi / d_min
    print(k_min, k_max)

    return k_min, k_max


def subset_data(
    subset_type,
    freqs_grid,
    vels_grid,
    curve_freqs,
    y_curve,
    polygon=None,
    k_limits=None,
    y_max=None,
    remove_artifact=False,
):
    """
    subset data, either using the polygon or the k limits.
    """
    residuals_freq = []
    residuals_grid = []
    quant_5 = []
    quant_95 = []
    for f in curve_freqs:
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
            # still subset for the aliasing limit

            if (y_curve[curve_freqs == f].values[0] >= min_2) and (
                y_curve[curve_freqs == f].values[0] <= max_2
            ):
                if remove_artifact:
                    res = list(
                        vels[(vels >= min_2) & (vels <= y_max)]
                        - y_curve[curve_freqs == f].values[0]
                    )
                else:
                    res = list(
                        vels[vels <= y_max] - y_curve[curve_freqs == f].values[0]
                    )
            else:
                res = []

        else:
            res = list(vels - y_curve[curve_freqs == f].values[0])
        residuals_freq += list(np.repeat(f, len(res)))
        residuals_grid += res

        if res:
            quant_5.append(np.quantile(res, 0.05))
            quant_95.append(np.quantile(res, 0.95))
        else:
            quant_5.append(None)
            quant_95.append(None)

    return residuals_freq, residuals_grid, quant_5, quant_95
