import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import re
import xarray as xr
from matplotlib.colors import LogNorm
from disba import PhaseDispersion

import sys

sys.path.append("../mcmc/src/")

# from plotting.plot_dispersion_curve import *

# import statsmodels.api as sm


def plot_inversion(file_name):
    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    # I don't think the .close() is necessary.
    input_ds.close()
    results_ds.close()

    plot_results(
        input_ds,
        results_ds,
        out_filename=file_name,
        # plot_prob_model=True,
    )


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    # file_name = "1773006102"
    file_name = "1773006274"
    # file_name = "1773006297"
    # file_name = "1773006314"
    # file_name = "1773006356"
    # file_name = "1773006377"
    # file_name = "1773006411"
    # file_name = "1773006603"
    # file_name = "1773006650"
    # file_name = "1773006661"
    # file_name = "1773006687"
    # file_name = "1773006713"
    # file_name = "1773006742"
    # file_name = "1773006772"
    # file_name = "1773006799"
    # file_name = "1773006832"
    # file_name = "1773006856"
    # file_name = "1773006902"

    plot_inversion(file_name)
