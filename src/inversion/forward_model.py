import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

np.complex_ = np.complex64


class Model:
    def __init__(
        self,
        param_bounds,
        n_layers: int,
        n_data: int,
        periods: np.ndarray,
        poisson_ratio: float,
        density_params: list,
    ):
        """
        :param n_layers: number of layers in the model.
        :param n_data: number of data collected
        :param freqs: frequencies at which data was collected.
        :param sigma_model: uncertainty in data
        :param param_bounds: array of (min, max, range) for each parameter. same length as n_params
        :param poisson_ratio: value for poisson's ratio used to approximate vel_p from vel_s
        :param density_params: birch params used to estimate the density profile of the model
        """
        self.n_layers = n_layers
        self.n_data = n_data
        self.periods = periods

        # used for computing model params
        self.poisson_ratio = poisson_ratio
        self.density_params = density_params

        # assemble model params
        self.param_bounds = param_bounds
        self.n_params = len(self.param_bounds)
        self.model_params = self.generate_model_params()

    @property
    def layer_bounds(self):
        return self.param_bounds[0]

    @property
    def sigma_model_bounds(self):
        return self.param_bounds[-1]

    @property
    def thickness(self):
        return self.model_params[: self.n_layers]

    @property
    def vel_s(self):
        return self.model_params[self.n_layers : 2 * self.n_layers]

    @property
    def vel_p(self):
        vp_vs = np.sqrt((2 - 2 * self.poisson_ratio) / (1 - 2 * self.poisson_ratio))
        vel_p = self.vel_s * vp_vs
        return vel_p

    @property
    def density(self):
        # *** don't recompute vel_p ***
        density = (self.vel_p - self.density_params[0]) / self.density_params[1]
        return density

    @property
    def sigma_model(self):
        return self.model_params[-1]

    @sigma_model.setter
    def sigma_model(self, sigma_model):
        self.model_params[-1] = sigma_model

    def get_thickness(self, model_params):
        return model_params[: self.n_layers]

    def get_vel_s(self, model_params):
        return model_params[self.n_layers : 2 * self.n_layers]

    def get_vel_p(self, vel_s):
        vp_vs = np.sqrt((2 - 2 * self.poisson_ratio) / (1 - 2 * self.poisson_ratio))
        vel_p = vel_s * vp_vs
        return vel_p

    def get_density(self, vel_p):
        density = (vel_p - self.density_params[0]) / self.density_params[1]
        return density

    @staticmethod
    def get_sigma_model(model_params):
        return model_params[-1]

    @staticmethod
    def assemble_param_bounds(bounds, n_layers):
        # reshape bounds to be the same shape as params
        param_bounds = np.concatenate(
            (
                [bounds["thickness"]] * n_layers,
                [bounds["vel_s"]] * n_layers,
                [bounds["vel_p"]] * n_layers,
                [bounds["density"]] * n_layers,
                [bounds["sigma_model"]],
            ),
            axis=0,
        )

        # add the range of the bounds to param_bounds as a third column (min, max, range)
        range = param_bounds[:, 1] - param_bounds[:, 0]
        param_bounds = np.column_stack((param_bounds, range))

        return param_bounds

    def generate_model_params(self):
        """
        generating initial params for new model.
        """
        model_params = np.random.uniform(
            self.param_bounds[:, 0], self.param_bounds[:, 1], self.n_params
        )

        return model_params

    def get_velocity_model(self, model_params):
        """
        not used for generalized inversion.
        reshape model params to be inputed into forward model PhaseDispersion.
        """
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        thickness = self.get_thickness(model_params)
        vel_s = self.get_vel_s(model_params)
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)

        return [thickness, vel_p, vel_s, density]

    def forward_model(self, model_params):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param model_params: model params to use to get phase dispersion
        """
        # get phase dispersion curve
        velocity_model = self.get_velocity_model(model_params)
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(self.periods, mode=0, wave="rayleigh")
            # ell = Ellipticity(*velocity_model.T)
            phase_velocity = pd_rayleigh.velocity

            return phase_velocity

        except (DispersionError, ZeroDivisionError) as e:
            # *** errors: ***
            # failed to find root for fundamental mode
            # division by zero
            raise e


class SyntheticModel(Model):
    def __init__(self, bounds, args):
        """
        Generate synthetic model.

        :param n_data: Number of observed data to simulate.
        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        """

        param_bounds = self.assemble_param_bounds(bounds, args["n_layers"])
        super().__init__(param_bounds, **args)  # generates model params and data

        # generate simulated observed data by adding Gaussian noise to true values.
        self.data_obs = self.data_true + self.sigma_model * np.random.randn(self.n_data)

    def generate_model_params(self):
        """
        generating true velocity model.
        """
        # generating initial model params. generate params until the forward model runs without error.
        valid_params = False

        while not valid_params:
            model_params = super().generate_model_params()
            try:
                # get the true data values for the true model
                self.data_true = self.forward_model(model_params)
                valid_params = True
            except (DispersionError, ZeroDivisionError):
                continue

        return model_params


class DataModel(Model):
    def __init__(self, data_path, args):
        """ """
        super().__init__(*args)

        periods, data_obs = self.read_observed_data(data_path)
        self.periods = periods
        self.data_obs = data_obs

    def read_observed_data(self, path):
        """
        read dispersion curve
        """
        names = ["frequency", "slowness", "std", "n_bins", "valid"]
        df = pd.read_csv(path, sep="\s+", names=names)

        periods = np.flip(1 / df["frequency"])
        phase_vel = 1 / df["slowness"]

        return periods, phase_vel


class ChainModel(Model):
    def __init__(self, beta, n_bins, *args):
        """
        :param beta: inverse temperature; larger values explore less of the parameter space,
            but are more precise; between 0 and 1
        :param n_bins: number of bins for histogram
        """

        super().__init__(*args)

        self.beta = beta

        # variables for storing and computing covariance matrix after burn-in
        self.rot_mat = np.eye(self.n_params)  # initialize rotation matrix
        self.n_cov = 0  # initialize the dividing number for covariance
        self.mean_model = np.zeros(self.n_params)
        self.mean_model_sum = np.zeros((self.n_params))

        self.cov_mat = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix
        self.cov_mat_sum = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix sum

        # acceptance ratio for each parameter
        self.swap_acc = 0
        self.swap_prop = 0

        # initialize histogram of model parameters
        self.n_bins = n_bins
        self.model_hist = np.zeros((self.n_params, n_bins + 1))

    def generate_model_params(self):
        """
        generating initial params for new model.
        """
        model_params = np.random.uniform(
            self.param_bounds[:, 0], self.param_bounds[:, 1], self.n_params
        )

    def get_optimization_model(self, param_bounds):
        # annealing schedule
        # starting temperature
        # >80-90% accepted initially

        results = {
            "temperature": [],
            "acc_rate": [],
            "misfit": [],
            "model": [
                [],
                [],
                [],
                [],
                [],
            ],
            "d_pred": [],
        }

        # random starting model
        # should already have initial model from init
        m = np.random.uniform(param_bounds[:, 0], param_bounds[:, 1])
        # E, _ = self.calc_E(m, d_obs, theta, sigma)

        """
        # temperature reduction factor
        T_0 = 1e2
        n_steps = 50
        temps = np.logspace(2, 0, 500)
        """

        T_0 = 100  # Initial Temp
        epsilon = 0.95  # Decayfactor of temperature
        ts = 300  # Number of temperature steps
        n_steps = 200  # Number of balancing steps at each temp

        temps = np.zeros(ts)
        temps[0] = T_0

        for k in range(1, ts):
            temps[k] = temps[k - 1] * epsilon

        results["acc_rate"] = []
        # reduce logarithmically
        for T in temps:
            results["acc_rate"].append([])
            # number of steps at this temperature
            for _ in range(n_steps):
                # perturb each parameter
                for param_ind in range(len(m)):
                    # cauchy distribution
                    eta = np.random.uniform(0, 1)

                    gamma = (T / T_0) * np.tan(np.pi * (eta - 0.5))
                    # m_new = m_try
                    m_new = m.copy()

                    perturb = gamma * scale_factor[param_ind]
                    m_new[param_ind] = m_new[param_ind] + perturb

                    E_new, d_new = calc_E(m_new, d_obs, theta, sigma)

                    delta_E = E_new - E
                    acc = False
                    if (m_new[param_ind] >= param_bounds[param_ind][0]) and (
                        m_new[param_ind] <= param_bounds[param_ind][1]
                    ):
                        if delta_E <= 0:
                            m = m_new.copy()
                            E = E_new
                            acc = True
                        else:
                            xi = np.random.uniform(0, 1)
                            if xi <= np.exp(-delta_E / T):
                                m = m_new.copy()
                                E = E_new
                                acc = True

                    results["acc_rate"][-1].append(acc)

            results["temperature"].append(T)
            results["model"].append(m)
            results["misfit"].append(delta_E)
            results["d_pred"].append(d_new)
            results["acc_rate"][-1] = np.sum(results["acc_rate"][-1]) / len(
                results["acc_rate"][-1]
            )

        return results, d_new, epsilon, n_steps

    def perturb_params(self, param_bounds):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
        """
        # should be validating params, generate until have valid values
        # get bounds in rotated space? validate in rotated space...

        # normalizing params
        norm_params = (self.model_params - self.param_bounds[:, 0]) / param_bounds[:, 2]
        # rotating params
        rotated_params = np.matmul(np.transpose(self.rot_mat), norm_params)

        # generate params to try; Cauchy proposal
        perturbed_rotated_params = rotated_params + (
            self.sigma_model * np.tan(np.pi * (np.random.rand(self.n_params) - 0.5))
        )

        # rotating back
        perturbed_norm_params = np.matmul(self.rot_mat, perturbed_rotated_params)
        # rescaling
        perturbed_params = param_bounds[:, 0] + (
            perturbed_norm_params * param_bounds[:, 2]
        )

        # validate params

        # boolean array of valid params
        # valid_params = (test_params >= self.param_bounds[:, 0]) & (
        #    test_params <= self.param_bounds[:, 1]
        # )

        # loop over params and perturb each individually
        for ind in np.arange(self.n_params):
            # calculate new likelihood
            try:
                test_params = self.model_params
                test_params[ind] = perturbed_params[ind]

                logL_new = self.get_likelihood(
                    test_params,
                )
            except (DispersionError, ZeroDivisionError):
                continue

            # Compute likelihood ratio in log space:
            dlogL = logL_new - self.logL
            if dlogL == 0:
                continue

            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                self.swap_acc += 1
                self.model_params[ind] = test_params[ind]
                self.logL = logL_new
            else:
                self.swap_prop += 1

    def get_derivatives(
        self,
        n_sizes,
        data_diff_bounds,
    ):
        """
        calculate the jacobian for the model

        :param n_sizes: number of step sizes to try
        :param phase_vel_diff_bounds:
        """
        # propose n_dm=50 step sizes. compute all for all params. find flat section and optimal derivative. add prior
        # estimate to make it stable finding where the derivative is flat to find best / stable value of the derivative
        # for the jacobian.

        # step size is scaled from the param range
        step_scales = np.linspace(0.1, 0.001, n_sizes)
        step_sizes = np.repeat(step_scales, self.n_params) * self.model_params[:, 2]

        model_derivatives = np.zeros((self.n_params, self.n_data, n_sizes))

        # estimate deriv for range of dm values
        for param_ind in range(self.n_params):
            model_pos = self.model_params + step_sizes[param_ind, :]
            model_neg = self.model_params - step_sizes[param_ind:, :]

            model_pos[param_ind] = model_pos[param_ind] + step_sizes[param_ind, :]

            try:
                data_pos = self.forward_model(model_pos)
                data_neg = self.forward_model(model_neg)

                # calculate the change in phase velocity over change in model param
                # unitless difference between positive and negative phase velocities
                data_diff = np.abs((data_pos - data_neg) / (data_pos + data_neg))

                # calculate centered derivative for values with reasonable differences(?)
                inds = (data_diff > data_diff_bounds[0]) & (
                    data_diff < data_diff_bounds[1]
                )
                model_derivatives[:, :, inds] = (data_pos - data_neg) / (
                    2 * step_sizes[param_ind, inds]
                )
            except (DispersionError, ZeroDivisionError) as e:
                pass

        return model_derivatives

    def get_jacobian(
        self,
    ):
        """
        finding the step size where the derivative is stable (flat)
        """
        n_sizes = 50
        size_scale = 1.5
        init_step_size_scale = 0.1
        phase_vel_diff_bounds = [1.0e-7, 5]

        model_derivatives = self.get_derivatives(
            n_sizes, size_scale, init_step_size_scale, phase_vel_diff_bounds
        )  # [param, deriv, data]

        Jac = np.zeros((self.n_data, self.n_params))

        # get indices of derivatives that are too small
        small_indices = []
        large_indices = []
        best_indices = []
        for s in range(n_sizes - 2):
            small_indices.append(
                np.any(np.abs(model_derivatives[:, s : s + 2, :]) < 1.0e-7)
            )
            large_indices.append(
                np.any(np.abs(model_derivatives[:, s : s + 2, :]) > 1.0e10)
            )
            # want three in a row
            # smallest difference between them?
            # absolute value of the sum of the left and right derivatives

            flatness = np.sum(model_derivatives[:, s : s + 2, :])
            best = np.argmin(flatness)
            best_indices.append(model_derivatives[:, best, :])

        Jac = model_derivatives[:, best_indices, :]

        return Jac

    def linearized_rotation(self, param_bounds):
        """
        making a linear approximation of the rotation matrix and variance for the params.

        :param variance: from the uniform distribution/ prior

        :return sigma_pcsd:
        """
        Jac = self.get_jacobian()
        # Scale columns of Jacobian for stability
        Jac = Jac * self.param_bounds[:, 2]  # multiplying by parameter range

        # Uniform bounded priors of width Δmi are approximated by taking C_p to be a diagonal matrix with
        cov_prior_inv = np.diag(self.n_params * [1 / param_bounds[:, 2]])

        # the data covariance matrix
        cov_data_inv = np.diag(self.n_data * [self.beta / self.sigma_data**2])

        cov_cur = (
            np.matmul(np.matmul(np.transpose(Jac), cov_data_inv), Jac) + cov_prior_inv
        )

        # parameter variance in PC space (?)
        rot_mat, s, _ = np.linalg.svd(cov_cur)
        sigma_model = 1 / (2 * np.sqrt(np.abs(s)))  # PC standard deviations

        return rot_mat, sigma_model

    def get_likelihood(self, data_obs):
        """
        :param model_params: params to calculate likelihood with
        """
        n_params = len(self.model_params)
        try:
            data_pred = self.forward_model(self.model_params)
            residuals = data_obs - data_pred

            sigma_model = self.get_sigma_model(self.model_params)

            # *** fix this
            logL = -(1 / 2) * n_params * np.log(sigma_model) - np.sum(
                residuals**2
            ) / (2 * sigma_model**2)

            return np.sum(logL)

        except (DispersionError, ZeroDivisionError) as e:
            raise e

    def update_model_hist(self):
        """
        updating the hist for this model, which stores parameter values from all the models
        """
        # The bins for this hist should be the param bounds
        for ind in range(self.n_params):
            counts, bins = np.histogram(x, n_bins)

            # getting bins for the param,
            # edge = self.bins[:, ind]
            # idx_diff = np.argmin(abs(edge - self.model_params[ind]))
            # self.model_hist[idx_diff, ind] += 1

    def update_rotation_matrix(self, burn_in):
        # for burn in period, update rotation matrix by linearization
        # after burn in, start saving samples in covariance matrix
        # after burn in (and cov mat stabilizes) start using cov mat to get rotation matrix
        # update covariance matrix

        if burn_in:
            # linearize
            rot_mat, sigma_model = self.lin_rot()
        else:
            rot_mat, sigma_model = self.update_covariance_matrix()

        self.rot_mat, self.sigma_model = rot_mat, sigma_model

    def update_covariance_matrix(self):
        """ """
        # normalizing
        normalized_model = (
            self.model_params - self.param_bounds[:, 0]
        ) / self.param_bounds[:, 2]

        self.mean_model_sum += normalized_model  # calculating the sum of mean
        self.n_cov += 1  # number of covariance matrices in the sum

        # *** validate this ****
        mean_model = self.mean_model_sum / self.n_cov

        self.cov_mat_sum = self.cov_mat_sum + np.outer(
            np.transpose(normalized_model - mean_model),
            normalized_model - mean_model,
        )

        # calculating covariance matrix from samples
        self.cov_mat = self.cov_mat_sum / self.n_cov

        # *** simplify? ***
        # dividing the covariance matrix by the auto-correlation of the params, and data
        for row in range(self.n_params):
            for col in range(self.n_params):
                self.cov_mat[row, col] /= np.sqrt(  # invalid scalar divide
                    self.cov_mat[row, row] * self.cov_mat[col, col]
                )

        rot_mat, s, _ = np.linalg.svd(
            self.cov_mat
        )  # rotate it to its Singular Value Decomposition
        sigma_model = np.sqrt(s)

        # s is the step size? sampling based on sigma_model

        return rot_mat, sigma_model
