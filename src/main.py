from plotting.inversion_plotting import *
from plotting.data_plotting import *

import asyncio
import numpy as np
from tests.test_inversion import *

import sys

sys.path.append("../mcmc/src/")
from inversion.data import SyntheticData, FieldData
from inversion.inversion import Inversion


if __name__ == "__main__":

    # max_path = "./data/WH01/max_files/WH01_fine.max"
    # txt_path = "./data/WH01/txt_files/WH01_curve_fine.txt"
    max_path = "./data/WH02/max_files/WH02_fine.max"
    txt_path = "./data/WH02/txt_files/WH02_curve_fine.txt"

    plot_geopsy_dispersion_curve(max_path, txt_path)
    plot_computed_dispersion_curve(max_path, f_min=2, f_max=10)
