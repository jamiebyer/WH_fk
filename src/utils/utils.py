import os
import datetime

import pandas as pd


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
