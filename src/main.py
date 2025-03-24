from processing.run_geopsy import *
from processing.run_obspy import *
from inversion.forward_model import *
from plotting.inversion_plotting import *

import numpy as np


def run_inversion(synthetic_data=True):
    # *** model generation / chain generation
    model_kwards = {
        "poisson_ratio": 0.265,
        "density_params": [540.6, 360.1],  # *** check units ***
        "n_layers": 10,
        "n_chains": 2,
        "beta_spacing_factor": 1.15,
        "model_variance": 12,
    }

    inversion_init_kwargs = (
        {
            "n_bins": 200,
            "n_burn": 10000,
            "n_keep": 2000,
            "n_rot": 40000,
        },
    )

    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }

    if synthetic_data:
        synthetic_model_kwargs = {
            "poisson_ratio": 0.265,
            "density_params": [540.6, 360.1],  # *** check units ***
            "n_data": 10,
            "n_layers": 3,
            "layer_bounds": [5e-3, 15e-3],  # km
            "vel_s_bounds": [2, 6],  # km/s
            "sigma_pd_bounds": [0, 1],
        }

        model = SyntheticModel()

    else:
        n_data, freqs, data_obs = read_observed_data()
        periods = np.flip(1 / freqs)
        data_obs = np.flip(data_obs)

        model = DataModel()

    # run inversionl
    inversion = Inversion(
        model,
        **data_kwargs,
        **inversion_init_kwargs,
    )

    # *** should any of those params just be in random walk?
    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(**inversion_run_kwargs)
    )


if __name__ == "__main__":
    run_inversion()
    # plot_observed_data()
