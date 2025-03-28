from inversion.forward_model import SyntheticModel, DataModel
from plotting.inversion_plotting import *
from inversion.inversion import Inversion
import asyncio
import numpy as np
from processing.run_geopsy import plot_max_file_curve


def run_inversion(synthetic_data=True):
    inversion_init_kwargs = {
        "n_bins": 200,
        "n_burn": 10000,
        "n_keep": 2000,
        "n_rot": 40000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }
    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }
    model_kwargs = {
        "n_layers": 2,
        "n_data": 20,
        "density_params": [540.6, 360.1],
        "poisson_ratio": 0.265,
    }

    bounds = {
        "thickness": [5e-3, 15e-3],  # km
        "vel_s": [2, 6],  # km/s
        "vel_p": [2, 6],  # km/s
        "density": [1, 1000],
        "sigma_model": [0, 1],
    }

    if synthetic_data:
        periods = np.flip(1 / np.logspace(-2, 2, model_kwargs["n_data"]))
        model_kwargs.update({"periods": periods})

        model = SyntheticModel(bounds, model_kwargs)

    else:

        model = DataModel(model_kwargs)

    # run inversion
    inversion = Inversion(
        model,
        bounds,
        **inversion_init_kwargs,
    )

    # *** should any of those params just be in random walk?
    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(**inversion_run_kwargs)
    )


if __name__ == "__main__":
    run_inversion()
    # plot_observed_data()
    # plot_max_file_curve("./data/WH01/WH01_fine_test.max")
