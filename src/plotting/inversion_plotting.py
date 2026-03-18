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
    # file_name = "1761019167"

    # file_name = "1761104879"
    # file_name = "1761104903"
    # file_name = "1761104952"
    # file_name = "1761105016"
    # file_name = "1761105071"
    # file_name = "1761105143"

    # file_name = "1761147668"
    # file_name = "1761147729"
    # file_name = "1761147814"
    # file_name = "1761147908"
    # file_name = "1761148148"
    # file_name = "1761149176"

    # file_name = "1761186443"
    # file_name = "1761186497"
    # file_name = "1761186642"
    # file_name = "1761186677"
    # file_name = "1761186768"
    # file_name = "1761186815"

    # file_name = "1761236411"
    # file_name = "1761236463"
    # file_name = "1761236497"
    # file_name = "1761236585"
    # file_name = "1761236678"
    # file_name = "1761236707"

    file_name = "1762299408"
    # file_name = "1762299506"
    plot_inversion(file_name)
