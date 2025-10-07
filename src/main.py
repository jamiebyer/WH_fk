from plotting.inversion_plotting import *
from plotting.data_plotting import *
from fk_processing.dispersion_curves import *

import asyncio
import numpy as np

# from tests.test_inversion import *

import sys

sys.path.append("../mcmc/src/")
from inversion.data import SyntheticData, FieldData
from inversion.inversion import Inversion


if __name__ == "__main__":

    # max_path = "./data/WH01/max_files/WH01_fine.max"
    # txt_path = "./data/WH01/txt_files/WH01_curve_fine.txt"

    max_path = "./data/WH02/max_files/WH02_fine.max"
    # txt_path = "./data/WH02/txt_files/WH02_curve_fine.txt"

    # max_path = "./capon-importedsignals.max"

    # plot_geopsy_dispersion_curve(max_path, txt_path)
    # plot_computed_dispersion_curve(max_path, f_min=-1, f_max=100000)
    # plot_computed_dispersion_curve(max_path, f_range=[[2.2, 6]])  # WH01
    # plot_computed_dispersion_curve(max_path, f_range=[[2, 3.7], [9.8, 10000]])  # WH02

    # plot_dispersion_curve_frequency(
    #     max_path, f_range=[[2.2, 6]], freq=6
    # )

    plot_dispersion_curve_frequency(
        max_path, f_range=[[2, 3.7], [9.8, 10000]], freq=10.5
    )

    # compare_dispersion_curves(max_path, txt_path, f_min=2.2, f_max=7.5)

    # run_geopsy()
