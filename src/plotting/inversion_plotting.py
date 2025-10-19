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

from plotting.plot_dispersion_curve import *

# import statsmodels.api as sm


def plot_inversion(file_name):
    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    # I don't think the .close() is necessary.
    input_ds.close()
    results_ds.close()

    # plot_results(input_ds, results_ds, out_filename=file_name, 
        # plot_prob_model=True,
    # )
    resulting_model_histogram(
        input_ds,
        results_ds,
        n_bins=200,
        save=True,
        plot_prob_model=True,
        plot_true_model=False,
    )


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    # run_inversion()

    # file_name = "1759624427"
    # file_name = "1759624943"
    # file_name = "1759625244"
    # file_name = "1759625320"

    # file_name = "1759674999"
    # file_name = "1759675114"
    file_name = "1759675206"
    # file_name = "1759675296"

    plot_inversion(file_name)
