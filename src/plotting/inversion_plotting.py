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

    # file_name = "1761018802"
    # file_name = "1761018940"
    # file_name = "1761019060"
    file_name = "1761019167"

    plot_inversion(file_name)
