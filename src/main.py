from plotting.inversion_plotting import *
from plotting.data_plotting import *
from fk_processing.dispersion_curves import *

import asyncio
import numpy as np

import sys

sys.path.append("../mcmc/src/")
# sys.path.append("./src/")
from inversion.data import SyntheticData, FieldData
from inversion.inversion import Inversion

import numpy as np

from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

# from plotting.plot_dispersion_curve import *

from fk_processing.dispersion_curves import compute_dispersion_curve

import xarray as xr


np.random.seed(0)


def setup_model(n_layers, site):
    # set up data and inversion params
    if site == "WH01":
        if n_layers == 1:
            proposal_width = {
                "depth": 0.03,
                "vel_s": [0.03, 0.15],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([0.001, 0.050]),  # km
                "vel_s": np.array([[0.100, 0.500], [0.300, 1.500]]),  # km/s
            }
        elif n_layers == 2:
            proposal_width = {
                "depth": [0.03, 0.05],
                "vel_s": [0.03, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.100], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.500], [0.200, 1.000], [0.750, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.03, 0.05, 0.05],
                "vel_s": [0.03, 0.03, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.010], [0.005, 0.150], [0.005, 0.250]]
                ),  # km
                "vel_s": np.array(
                    [[0.100, 0.500], [0.200, 1.000], [0.200, 2.000], [1.500, 4.000]]
                ),  # km/s
            }
        elif n_layers == 4:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.100], [0.005, 0.150], [0.005, 0.250], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 0.500],
                        [0.200, 1.000],
                        [0.200, 2.000],
                        [1.500, 4.000],
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
    elif site == "WH02":
        if n_layers == 1:
            proposal_width = {
                "depth": 0.03,
                "vel_s": [0.03, 0.15],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([0.001, 0.050]),  # km
                "vel_s": np.array([[0.100, 0.500], [0.300, 1.500]]),  # km/s
            }
        elif n_layers == 2:
            proposal_width = {
                "depth": [0.03, 0.05],
                "vel_s": [0.03, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.100], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.500], [0.200, 1.000], [0.750, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.03, 0.05, 0.05],
                "vel_s": [0.03, 0.03, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.010], [0.005, 0.150], [0.005, 0.250]]
                ),  # km
                "vel_s": np.array(
                    [[0.100, 0.500], [0.200, 1.000], [0.200, 2.000], [1.500, 4.000]]
                ),  # km/s
            }
        elif n_layers == 4:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.100], [0.005, 0.150], [0.005, 0.250], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 0.500],
                        [0.200, 1.000],
                        [0.200, 2.000],
                        [1.500, 4.000],
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }

    model_params_kwargs = {
        "n_layers": n_layers,
        "vpvs_ratio": 1.75,
        "param_bounds": bounds,
        "proposal_width": proposal_width,
    }
    # model params
    model_params = DispersionCurveParams(**model_params_kwargs)

    return model_params


def basic_inversion(
    n_layers, sample_prior, set_starting_model, out_filename="", site=None
):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """

    data = setup_data(site)
    model_params = setup_model(n_layers, site)

    inversion_init_kwargs = {
        # "n_burn": 300000,
        "n_burn": 200000,
        "n_chunk": 500,
        # "n_mcmc": 1000000,
        "n_mcmc": 500000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "individual_acceptance": True,
        "out_filename": out_filename,
    }

    model_kwargs = {"sigma_data": data.sigma_data}

    # model_kwargs = {"sigma_data": stds}  # sigma_data * data.data_obs}

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    return inversion, model_params


def run_inversion():
    """
    - Run with sampling prior. Run with setting the starting model, run without.
        - Run with 1 layer, 2 layers.
        - Run with low noise, medium noise, high noise.
    """

    sample_prior = False
    set_starting_model = False
    rotate = False
    n_layers = 3

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
        # site="WH01",
        site="WH02",
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        rotate_params=rotate,
    )


def plot_dispersion_curve():

    # max_path = "./data/WH01/max_files/WH01_fine.max"
    # txt_path = "./data/WH01/txt_files/WH01_curve_fine.txt"

    max_path = "./data/WH02/max_files/WH02_fine.max"
    # txt_path = "./data/WH02/txt_files/WH02_curve_fine.txt"

    # max_path = "./capon-importedsignals.max"

    # plot full curve
    """
    # plot_computed_dispersion_curve(max_path, f_range=[[2.2, 7]], err_thresh=6)  # WH01
    plot_computed_dispersion_curve(
        max_path,
        f_range=[[2, 3.7], [6.5, 13]],
        freq_outliers=WH02_freqs,
        vel_outliers=WH02_vels,
    )  # WH02
    """

    # plot individual frequencies
    # plot_dispersion_curve_frequency(
    #     max_path, f_range=[[2.2, 7]], freq=2.2
    # )
    # plot_dispersion_curve_frequency(
    #    max_path, f_range=[[2, 3.7], [9.8, 13]], freq=9.5
    # )

    # compare_dispersion_curves(max_path, txt_path, f_min=2.2, f_max=7.5)

    # plot_site_maps()
    pass


if __name__ == "__main__":

    # run_inversion()
    # ambient_noise_data(site="WH01")
    # ambient_noise_data(site="WH02")

    # run_inversion()

    # plot final paper results
    file_name = "1758652456"

    input_path = (
        "/home/jbyer/Documents/uoc/repos/mcmc/results/inversion/input-"
        + file_name
        + ".nc"
    )
    results_path = (
        "/home/jbyer/Documents/uoc/repos/mcmc/results/inversion/results-"
        + file_name
        + ".nc"
    )

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    plot_full_results(input_ds, results_ds, n_bins=100, save=True)
