import asyncio
import numpy as np
import xarray as xr

import sys

sys.path.append("../mcmc/src/")
from inversion.data import SyntheticData, FieldData
from inversion.inversion import Inversion
from inversion.model_params import DispersionCurveParams

sys.path.append("./src/")
from plotting.well_holes import *
from plotting.inversion_plotting import *
from plotting.data_plotting import *
from processing.dispersion_curves import *



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
                "depth": [0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.050], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.750], [0.200, 2.000], [0.500, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.05, 0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.050], [0.005, 0.250], [0.005, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [[0.100, 0.750], [0.200, 2.000], [0.200, 2.000], [1.500, 4.000]]
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
                    [[0.005, 0.050], [0.005, 0.150], [0.005, 0.500], [0.100, 0.600]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 0.500], 
                        [0.200, 2.000], 
                        [0.200, 2.000], 
                        [0.200, 2.000],
                        [1.500, 3.000],
                    ]
                ),  # km/s
            }
        elif n_layers == 5:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [
                        [0.005, 0.050],
                        [0.005, 0.150],
                        [0.005, 0.400],
                        [0.100, 0.500],
                        [0.100, 0.600],
                    ]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 0.500],
                        [0.200, 2.000], 
                        [0.200, 2.000], 
                        [0.200, 2.000], 
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
    elif site == "WH02":
        if n_layers == 1:
            proposal_width = {
                "depth": 0.05,
                "vel_s": [0.05, 0.15],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([0.001, 0.040]),  # km
                "vel_s": np.array([[0.100, 0.750], [0.300, 1.500]]),  # km/s
            }
        elif n_layers == 2:
            proposal_width = {
                "depth": [0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.050], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.750], [0.200, 1.000], [0.750, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.05, 0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.075], [0.005, 0.125], [0.005, 0.400]]
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
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
        elif n_layers == 5:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
    elif site == "WH03":
        if n_layers == 1:
            proposal_width = {
                "depth": 0.05,
                "vel_s": [0.05, 0.15],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([0.001, 0.040]),  # km
                "vel_s": np.array([[0.100, 0.750], [0.300, 1.500]]),  # km/s
            }
        elif n_layers == 2:
            proposal_width = {
                "depth": [0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.050], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.750], [0.200, 1.000], [0.750, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.05, 0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.075], [0.005, 0.125], [0.005, 0.400]]
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
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
        elif n_layers == 5:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
    elif site == "WH04":
        if n_layers == 1:
            proposal_width = {
                "depth": 0.05,
                "vel_s": [0.05, 0.15],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([0.001, 0.040]),  # km
                "vel_s": np.array([[0.100, 0.750], [0.300, 1.500]]),  # km/s
            }
        elif n_layers == 2:
            proposal_width = {
                "depth": [0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.10],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array([[0.005, 0.050], [0.005, 0.200]]),  # km
                "vel_s": np.array(
                    [[0.100, 0.750], [0.200, 1.000], [0.750, 3.000]]
                ),  # km/s
            }
        elif n_layers == 3:
            proposal_width = {
                "depth": [0.05, 0.05, 0.05],
                "vel_s": [0.05, 0.05, 0.05, 0.05],
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.075], [0.005, 0.125], [0.005, 0.400]]
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
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
                        [1.500, 4.000],
                    ]
                ),  # km/s
            }
        elif n_layers == 5:
            proposal_width = {
                "depth": 0.05,
                "vel_s": 0.05,
            }  # fractional step size (multiplied by param bounds width)

            # set up data and inversion params
            bounds = {
                "depth": np.array(
                    [[0.005, 0.125], [0.005, 0.125], [0.005, 0.400], [0.005, 0.400], [0.100, 0.400]]
                ),  # km
                "vel_s": np.array(
                    [
                        [0.100, 1.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000],
                        [0.200, 2.000], 
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

    data, _, _ = setup_data(site)
    model_params = setup_model(n_layers, site)

    inversion_init_kwargs = {
        "n_burn": 500000,
        # "n_burn": 200000,
        "n_chunk": 500,
        "n_mcmc": 1500000,
        # "n_mcmc": 500000,
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
    n_layers = 5

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
        site="WH04",
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        rotate_params=rotate,
    )


if __name__ == "__main__":

<<<<<<< HEAD
    run_inversion()
    
=======
    # max_path = "./results/fk/WH04/1C/conventional-WH04-longest-default08.max"
    # max_path = "./results/fk/WH04/2C/conventionaltransverse-WH04-longest-default03.max"
    # max_path = "./results/WH01/3C/rtbf-WH01-test01.max"
    # plot_computed_dispersion_curve_curr(max_path)

    max_path = "./results/fk/final/conventional-WH01_3C_split-default08.max"
    curve_path = "./results/curves/curve-WH01-1C.csv"
    # plot_slowness(max_path, curve_path)
    # [plot_dispersion_curve_frequency(max_path, freq=f) for f in np.arange(2, 12, 0.5)]
    # plot_dispersion_curve_frequency(max_path, freq=6)

    max_paths = [
        "./results/fk/WH04/1C/conventional-WH04-longest-default06.max",
        "./results/fk/WH04/2C/conventionaltransverse-WH04-longest-default03.max",
    ]
    # plot_double_dispersion_curves(max_paths)

    # max_paths = [max_path.replace("4", str(i)) for i in np.arange(4, 10)]
    # plot_multiple_dispersion_curves(max_paths)

    # plot_array_response("WH01")
    # plot_raw_data("WH01")
    # slice_noise_data()
    # plot_array_layout()

    # well_hole_plotting("WH02")

    plot_curve_picking()

    # run_inversion()
>>>>>>> 9302e91da1f04f2deafb4e4532b8b836a863d706
