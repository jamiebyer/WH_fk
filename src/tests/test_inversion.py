import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import sys

sys.path.append("./src/")

from processing.dispersion_curves import compute_dispersion_curve

sys.path.append("../mcmc/src/")
from inversion.data import FieldData
from inversion.model import Model
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion


np.random.seed(0)


def setup_test_model(n_layers):
    # set up example data
    proposal_width = {
        "depth": [0.20],
        "vel_s": [0.20],
    }  # fractional step size (multiplied by param bounds width)

    # set up data and inversion params
    bounds = {
        "depth": np.array([0.001, 0.3]),  # km
        "vel_s": np.array([[0.100, 0.500], [0.300, 1.000], [0.750, 2.000]]),  # km/s
        # "vel_s": np.array([0.100, 2.000]),  # km/s
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


def basic_inversion(n_layers, noise, sample_prior, set_starting_model, out_filename=""):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """
    sigma_data = noise

    model_params = setup_test_model(n_layers)

    max_path = "./data/WH01/max_files/WH01_fine.max"
    # txt_path = "./data/WH01/txt_files/WH01_curve_fine.txt"
    # max_path = "./data/WH02/max_files/WH02_fine.max"
    # txt_path = "./data/WH02/txt_files/WH02_curve_fine.txt"

    (
        _,
        _,
        freqs,
        phase_vels,
        stds,
    ) = compute_dispersion_curve(max_path, f_min=2, f_max=7)

    periods = np.flip(1 / freqs)
    phase_vels = np.flip(phase_vels / 1000)
    stds = np.flip(stds / 1000)
    data = FieldData(periods, phase_vels, stds)

    inversion_init_kwargs = {
        "n_burn": 100000,
        "n_chunk": 500,
        "n_mcmc": 500000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "out_filename": out_filename,
    }

    model_kwargs = {"sigma_data": 0.05}  # sigma_data * data.data_obs}

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    return inversion, model_params


def test_run_inversions():
    """
    - Run with sampling prior. Run with setting the starting model, run without.
        - Run with 1 layer, 2 layers.
        - Run with low noise, medium noise, high noise.
    """
    sample_prior = False
    set_starting_model = False
    rotate = False
    n_layers = 2
    noise = 0.05  # 0.02 # 0.05 # 0.1

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        noise=noise,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        rotate_params=rotate,
    )
