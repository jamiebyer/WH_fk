from processing.run_geopsy import *
from processing.run_obspy import *
from inversion.forward_model import SyntheticModel, DataModel
from plotting.inversion_plotting import *
from inversion.inversion import Inversion
import asyncio
import numpy as np


def run_inversion(synthetic_data=True):
    inversion_init_kwargs = {
        "n_bins": 200,
        "n_burn": 10000,
        "n_keep": 2000,
        "n_rot": 40000,
    }
    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }
    model_kwards = {
        "poisson_ratio": 0.265,
        "density_params": [540.6, 360.1],
        "n_layers": 2,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }

    if synthetic_data:
        model_kwargs = model_kwargs.update(
            {
                "n_data": 10,
                "layer_bounds": [5e-3, 15e-3],  # km
                "vel_s_bounds": [2, 6],  # km/s
                "sigma_pd_bounds": [0, 1],
            }
        )

        model = SyntheticModel(model_kwargs)

    else:
        # *** model generation / chain generation

        n_data, freqs, data_obs = read_observed_data()
        periods = np.flip(1 / freqs)
        data_obs = np.flip(data_obs)

        model = DataModel(model_kwards)

    # should be passing param bounds...
    # unless that's on model...

    # run inversionl
    inversion = Inversion(
        model,
        **inversion_init_kwargs,
    )

    # *** should any of those params just be in random walk?
    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(**inversion_run_kwargs)
    )


if __name__ == "__main__":
    # run_inversion()
    plot_observed_data()
