import numpy as np
import matplotlib.pyplot as plt


def calc_E(m, d_obs, theta, sigma):
    d_new = ref_coeff(m, theta)
    E = np.sum((d_obs - d_new) ** 2) / (2 * sigma**2)
    return E, d_new


def model_setup():
    # m = [v_p, v_s, rho, alpha_p, alpha_s]

    m_t = [1800, 400, 1850, 0.2, 0.5]
    param_bounds = np.array(
        [
            [1500, 2000],
            [0, 1000],
            [1200, 2200],
            [0, 1],
            [0, 1],
        ]
    )

    scale_factor = (1 / 30) * np.array(param_bounds[:, 1] - param_bounds[:, 0])

    theta = np.arange(0, 91)
    sigma = 0.02

    # create noisy data
    d_t = ref_coeff(m_t, theta)
    d_obs = d_t + np.random.normal(loc=0, scale=sigma, size=len(d_t))

    return (
        m_t,
        d_t,
        d_obs,
        theta,
        sigma,
        scale_factor,
        param_bounds,
    )


def get_SA_optimization_model(
    d_obs,
    param_bounds,
    theta,
    sigma,
):

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
    m = np.random.uniform(param_bounds[:, 0], param_bounds[:, 1])
    E, _ = calc_E(m, d_obs, theta, sigma)

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

        results["model"][0].append(m[0])
        results["model"][1].append(m[1])
        results["model"][2].append(m[2])
        results["model"][3].append(m[3])
        results["model"][4].append(m[4])

        results["misfit"].append(delta_E)
        results["d_pred"].append(d_new)
        results["acc_rate"][-1] = np.sum(results["acc_rate"][-1]) / len(
            results["acc_rate"][-1]
        )

    return results, d_new, epsilon, n_steps


def get_SA_mcmc_model(m_init, n_samples, scale_factor, theta, sigma):
    results = {
        "model": [
            [],
            [],
            [],
            [],
            [],
        ],
        "misfit": [],
        "acc": [],
        "d_pred": [],
    }

    # uniform prior and
    # symmetric proposal distribution

    # acceptance is min{1, L(m')/L(m)}

    m = m_init
    E, _ = calc_E(m, d_obs, theta, sigma)
    for _ in range(n_samples):
        # perturb each parameter
        m_try = m
        for param_ind in range(len(m)):
            # cauchy distribution
            eta = np.random.uniform(0, 1)

            gamma = np.tan(np.pi * (eta - 0.5))
            m_new = m_try.copy()

            m_new[param_ind] = m_try[param_ind] + gamma * scale_factor[param_ind]

            E_new, d_new = calc_E(m_new, d_obs, theta, sigma)

            delta_E = E_new - E
            acc = False
            if (
                m_new[param_ind] >= param_bounds[param_ind][0]
                and m_new[param_ind] <= param_bounds[param_ind][1]
            ):
                if delta_E <= 0:
                    m = m_new.copy()
                    E = E_new
                    acc = True
                else:
                    xi = np.random.uniform(0, 1)
                    if xi <= np.exp(-delta_E):
                        m = m_new.copy()
                        E = E_new
                        acc = True

            results["model"][0].append(m[0])
            results["model"][1].append(m[1])
            results["model"][2].append(m[2])
            results["model"][3].append(m[3])
            results["model"][4].append(m[4])

            results["misfit"].append(delta_E)
            results["acc"].append(acc)
            results["d_pred"].append(d_new)

    return results


### PLOTS


def plot_metrics(temps, acc_rate, misfit, d_obs, d_new, theta, epsilon, n_steps):
    plt.subplot(1, 3, 1)

    plt.plot(temps, acc_rate)
    plt.xscale("log")
    # plt.title("acceptance rate")
    plt.xlabel("temperature")
    plt.ylabel("acceptance rate")
    plt.xlim([temps[0], temps[-1]])

    plt.subplot(1, 3, 2)

    plt.plot(temps, misfit)
    plt.xlabel("temperature")
    plt.ylabel("misfit")
    plt.xscale("log")
    plt.xlim([temps[0], temps[-1]])

    plt.subplot(1, 3, 3)
    # plot annealing schedule
    plt.plot(np.flip(np.arange(len(temps))), temps)
    plt.ylabel("temperature")
    plt.xlabel("temperature step")
    plt.yscale("log")

    plt.suptitle(
        "initial temp: "
        + str(np.round(temps[0]))
        + ", epsilon: "
        + str(epsilon)
        + ", n temp steps: "
        + str(len(temps))
        + ", n perturb steps: "
        + str(n_steps)
    )

    plt.tight_layout()
    plt.show()


def plot_model_params(m, m_t, temperature, theta, d_pred, d_obs, d_true):
    # [v_p, v_s, rho, alpha_p, alpha_s]
    plt.subplot(3, 2, 1)

    plt.plot(temperature, m[0])
    plt.axhline(y=m_t[0], c="black")
    plt.ylabel("v_p (m/s)")
    plt.xscale("log")
    plt.xlabel("temperature")
    plt.gca().invert_xaxis()

    plt.subplot(3, 2, 2)

    plt.plot(temperature, m[1])
    plt.axhline(y=m_t[1], c="black")
    plt.ylabel("v_s (m/s)")
    plt.xscale("log")
    plt.xlabel("temperature")
    plt.gca().invert_xaxis()

    plt.subplot(3, 2, 3)

    plt.plot(temperature, m[2])
    plt.axhline(y=m_t[2], c="black")
    plt.ylabel("rho (kg/m^3)")
    plt.xscale("log")
    plt.xlabel("temperature")
    plt.gca().invert_xaxis()

    plt.subplot(3, 2, 4)

    plt.plot(temperature, m[3])
    plt.axhline(y=m_t[3], c="black")
    plt.ylabel("alpha_p")
    plt.xscale("log")
    plt.xlabel("temperature")
    plt.gca().invert_xaxis()

    plt.subplot(3, 2, 5)

    plt.plot(temperature, m[4])
    plt.axhline(y=m_t[4], c="black")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("temperature")
    plt.ylabel("alpha_s")

    plt.subplot(3, 2, 6)
    plt.plot(theta, d_true, c="black", label="true")
    plt.plot(theta, np.array(d_pred[-1]), label="pred")
    plt.scatter(theta, d_obs, c="black", s=6, label="obs")
    plt.ylabel("d pred")
    plt.xlabel("theta")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_model_params_mcmc(m, m_t, theta, d_pred, d_obs, d_true):
    # [v_p, v_s, rho, alpha_p, alpha_s]
    plt.subplot(3, 2, 1)

    plt.plot(m[0])
    plt.axhline(y=m_t[0], c="black")
    plt.ylabel("v_p (m/s)")
    plt.xlabel("step")

    plt.subplot(3, 2, 2)

    plt.plot(m[1])
    plt.axhline(y=m_t[1], c="black")
    plt.ylabel("v_s (m/s)")
    plt.xlabel("step")

    plt.subplot(3, 2, 3)

    plt.plot(m[2])
    plt.axhline(y=m_t[2], c="black")
    plt.ylabel("rho (kg/m^3)")
    plt.xlabel("step")

    plt.subplot(3, 2, 4)

    plt.plot(m[3])
    plt.axhline(y=m_t[3], c="black")
    plt.ylabel("alpha_p")
    plt.xlabel("step")

    plt.subplot(3, 2, 5)

    plt.plot(m[4])
    plt.axhline(y=m_t[4], c="black")
    plt.xlabel("step")
    plt.ylabel("alpha_s")

    plt.tight_layout()
    plt.show()


def plot_model_hists(m, m_t, param_bounds):
    # [v_p, v_s, rho, alpha_p, alpha_s]
    plt.subplot(5, 1, 1)

    plt.hist(m[0], bins=50, density=True)
    plt.axvline(x=m_t[0], c="black")
    plt.xlabel("v_p (m/s)")
    plt.ylabel("probability\ndensity")
    # plt.xlim(param_bounds[0])

    plt.subplot(5, 1, 2)

    plt.hist(m[1], bins=50, density=True)
    plt.axvline(x=m_t[1], c="black")
    plt.xlabel("v_s (m/s)")
    plt.ylabel("probability\ndensity")
    # plt.xlim(param_bounds[1])

    plt.subplot(5, 1, 3)

    plt.hist(m[2], bins=50, density=True)
    plt.axvline(x=m_t[2], c="black")
    plt.xlabel("rho (kg/m^3)")
    plt.ylabel("probability\ndensity")
    # plt.xlim(param_bounds[2])

    plt.subplot(5, 1, 4)

    plt.hist(m[3], bins=50, density=True)
    plt.axvline(x=m_t[3], c="black")
    plt.xlabel("alpha_p")
    plt.ylabel("probability\ndensity")
    # plt.xlim(param_bounds[3])

    plt.subplot(5, 1, 5)

    plt.hist(m[4], bins=50, density=True)
    plt.axvline(x=m_t[4], c="black")
    plt.xlabel("alpha_s")
    plt.ylabel("probability\ndensity")
    # plt.xlim(param_bounds[4])

    plt.tight_layout()
    plt.show()


def plot_data_predictions(results_opt, results_mcmc, d_obs, d_true, theta):
    plt.subplot(1, 2, 1)

    colors = plt.cm.jet(
        # np.linspace(0, 1, len(results_opt["temperature"])),
        results_opt["temperature"]
    )
    # colors = np.flip(colors)

    for ind in range(len(results_opt["temperature"])):
        col = colors[ind]
        col[3] = 0.4
        plt.plot(
            theta,
            np.array(results_opt["d_pred"]).T[:, ind],
            # label=np.array(results_opt["temperature"])[ind],
            c=col,
        )

    plt.scatter(theta, d_obs, c="black", zorder=2, label="obs", s=8)
    plt.plot(theta, d_true, c="black", label="true")
    plt.legend()
    plt.ylim([0.3, 1.05])
    plt.xlabel("theta (deg)")
    plt.ylabel("d")

    plt.subplot(1, 2, 2)
    inds = (
        np.random.choice(
            np.array(results_mcmc["d_pred"]).shape[0],
            len(results_opt["temperature"]),
        ),
    )
    subset = np.array(results_mcmc["d_pred"]).T[:, inds]

    plt.plot(
        theta,
        np.squeeze(subset),
        c=(0.8, 0.8, 0.8, 0.8),
        # label="subset",
    )
    plt.scatter(theta, d_obs, c="black", zorder=2, label="obs", s=8)
    plt.plot(theta, d_true, c="black", label="true")
    plt.ylim([0.3, 1.05])
    plt.xlabel("theta (deg)")
    # plt.ylabel("d")
    plt.legend()

    plt.tight_layout()
    plt.show()


m_t, d_t, d_obs, theta, sigma, scale_factor, param_bounds = model_setup()

results_opt, d_new, epsilon, n_steps = get_SA_optimization_model(
    d_obs,
    param_bounds,
    theta,
    sigma,
)


"""
plot_metrics(
    results_opt["temperature"],
    results_opt["acc_rate"],
    results_opt["misfit"],
    d_obs,
    d_new,
    theta,
    epsilon,
    n_steps,
)
"""
"""
plot_model_params(
    results_opt["model"],
    m_t,
    results_opt["temperature"],
    theta,
    results_opt["d_pred"],
    d_obs,
    d_t,
)
"""
n_samples = 35000

results_mcmc = get_SA_mcmc_model(
    [
        results_opt["model"][0][-1],
        results_opt["model"][1][-1],
        results_opt["model"][2][-1],
        results_opt["model"][3][-1],
        results_opt["model"][4][-1],
    ],
    n_samples,
    scale_factor,
    theta,
    sigma,
)
"""
plot_model_hists(
    results_mcmc["model"],
    m_t,
    param_bounds,
)
"""
"""
plot_model_params_mcmc(
    results_mcmc["model"],
    m_t,
    theta,
    results_mcmc["d_pred"],
    d_obs,
    d_t,
)
"""
plot_data_predictions(results_opt, results_mcmc, d_obs, d_t, theta)
