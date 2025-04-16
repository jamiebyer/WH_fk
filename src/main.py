from inversion.forward_model import SyntheticData, FieldData
from plotting.inversion_plotting import *
from inversion.inversion import Inversion
import asyncio
import numpy as np
from processing.run_geopsy import plot_max_file_curve


def run_inversion(synthetic_data=True):
    np.random.seed(0)

    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
        # "sigma_model": [0.01, 0.3],
    }
    model_kwargs = {
        "n_layers": 2,
        "sigma_model": 0.005,
        "poisson_ratio": 0.265,
        "param_bounds": bounds,
    }
    inversion_init_kwargs = {
        "n_bins": 200,
        "n_burn": 10000,
        "n_keep": 100,
        "n_rot": 40000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }
    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }

    if synthetic_data:
        n_data = 50
        periods = np.flip(1 / np.logspace(-2, 2, n_data))
        sigma_data = 0.005
        data_kwargs = {
            "thickness": [0.03],
            "vel_s": [0.4, 1.5],
            "vel_p": [1.6, 2.5],
            "density": [2.0, 2.5],
        }
        data = SyntheticData(periods, sigma_data, **data_kwargs)

    else:
        path = ""
        data = FieldData(path)

    # run inversion
    inversion = Inversion(
        data,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    inversion.random_walk(**inversion_run_kwargs)
    # asyncio.get_event_loop().run_until_complete(
    #    inversion.random_walk(**inversion_run_kwargs)
    # )


if __name__ == "__main__":
    in_path = "./results/inversion/results1744759431.nc"

    # run_inversion()
    # plot_starting_model()
    plot_optimized_model()

    # plot_pred_vs_obs(in_path)
    plot_inversion_results_param_prob(in_path)
    # plot_inversion_results_param_time(in_path)
    plot_inversion_results_logL(in_path)
