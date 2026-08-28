import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling_parameters(site, remove_artifact, remove_outliers, fig, ax):
    # fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 14), sharex=True)

    # read in individual fitting parameters
    path = (
        "./results/curve_fitting/remove_artifact_"
        + str(remove_artifact)
        + "/individual_fitting/"
        + site
        + "-asym-laplace-params.csv"
    )
    df = pd.read_csv(path)

    freqs = df["freqs"].values
    lambd = df["lambd"].values
    kappa = df["kappa"].values

    # read in smoothed scaling params
    path = (
        "./results/curve_fitting/scaling_params/"
        + site
        + "_scaling_params_"
        + str(remove_artifact)
        + ".csv"
    )
    df = pd.read_csv(path)
    scaling_params_freqs = df["freq"].values
    spread = df["spread"].values
    ratio = df["ratio"].values
    smoothed_spread = df["smoothed_spread"].values
    smoothed_ratio = df["smoothed_ratio"].values

    # multiply by fit scaling...

    # remove outliers
    if remove_outliers:
        lambd_inds = lambd <= 200
        if site == "WH03":
            kappa_inds = (kappa <= 5) & (1 / ratio <= 15)
        elif site == "WH04":
            kappa_inds = (kappa <= 5) & (1 / ratio <= 0.7)
        else:
            kappa_inds = kappa <= 5
    else:
        lambd_inds = np.full(len(lambd), True)
        kappa_inds = np.full(len(kappa), True)

    # get correlation coefficients
    spread_coef = np.corrcoef(lambd[lambd_inds], 1 / spread[lambd_inds])[0, 1]
    # skew_coef = np.corrcoef(kappa[kappa_inds], ratio[kappa_inds])[0, 1]
    skew_coef = np.corrcoef(kappa[kappa_inds], 1 / ratio[kappa_inds])[0, 1]
    # skew_coef = np.corrcoef(kappa[kappa_inds], spread[kappa_inds])[0, 1]
    """
    smoothed_spread_coef = np.corrcoef(
        lambd[lambd_inds], 1 / smoothed_spread[lambd_inds]
    )[0, 1]
    smoothed_skew_coef = np.corrcoef(kappa[kappa_inds], smoothed_ratio[kappa_inds])[
        0, 1
    ]
    """
    # plot spread and lambda
    # ax[0].scatter(freqs, lambd, label="lambda", c="blue", alpha=0.85)
    ax[0].scatter(
        freqs[lambd_inds], lambd[lambd_inds], label="lambda", c="blue", alpha=0.85
    )
    ax[0].set_ylabel("lambda")
    ax[0].legend(loc=(0.02, 0.85))

    ax_spread = ax[0].twinx()
    ax_spread.scatter(
        scaling_params_freqs, 1 / spread, label="1/spread", c="orange", alpha=0.85
    )
    ax_spread.plot(
        scaling_params_freqs, 1 / smoothed_spread, label="1/smoothed_spread", c="brown"
    )
    ax_spread.set_ylabel("1/spread")
    ax_spread.legend(loc=(0.02, 0.65))

    ax[0].set_title(
        str(
            np.round(spread_coef, 2)
        )  #  + ", " + str(np.round(smoothed_spread_coef, 2))
    )
    # plot ratio and kappa
    # ax[1].scatter(freqs, kappa, label="kappa", c="blue", alpha=0.85)
    ax[1].scatter(
        freqs[kappa_inds], kappa[kappa_inds], label="kappa", c="blue", alpha=0.85
    )
    ax[1].set_ylabel("kappa")
    ax[1].legend(loc=(0.02, 0.85))

    ax_ratio = ax[1].twinx()
    # ax_ratio.scatter(
    #     scaling_params_freqs, spread, label="spread", c="orange", alpha=0.85
    # )
    # ax_ratio.plot(
    #     scaling_params_freqs, smoothed_spread, label="smoothed_spread", c="brown"
    # )
    # ax_ratio.scatter(scaling_params_freqs, ratio, label="ratio", c="orange", alpha=0.85)
    # ax_ratio.plot(
    #     scaling_params_freqs, smoothed_ratio, label="smoothed_ratio", c="brown"
    # )
    ax_ratio.scatter(
        scaling_params_freqs[kappa_inds],
        1 / ratio[kappa_inds],
        label="1/ratio",
        c="orange",
        alpha=0.85,
    )
    ax_ratio.plot(
        scaling_params_freqs[kappa_inds],
        1 / smoothed_ratio[kappa_inds],
        label="1/smoothed_ratio",
        c="brown",
    )
    # ax_ratio.set_ylabel("spread")
    # ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylabel("1/ratio")
    ax_ratio.legend(loc=(0.02, 0.65))

    ax[1].set_title(
        str(np.round(skew_coef, 2))  # + ", " + str(np.round(smoothed_skew_coef, 2))
    )

    ax[1].set_xlabel("Frequency (Hz)")

    """
    plt.suptitle(
        site
        + ", remove artifact: "
        + str(remove_artifact)
        + ", remove outliers: "
        + str(remove_outliers)
    )
    plt.show()
    
    plt.savefig(
        "./figures/curve_fitting/scaling_params/params-"
        + str(remove_artifact)
        + "-"
        + str(remove_outliers)
        + "-"
        + site
        + ".png"
    )
    """


def plot_scaling_parameters_v2(site, remove_artifact, remove_outliers):
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 14), sharex=True)

    # read in individual fitting parameters
    path = (
        "./results/curve_fitting/remove_artifact_"
        + str(remove_artifact)
        + "/individual_fitting/"
        + site
        + "-asym-laplace-params.csv"
    )
    df = pd.read_csv(path)

    freqs = df["freqs"].values
    lambd = df["lambd"].values
    kappa = df["kappa"].values

    # read in smoothed scaling params
    path = (
        "./results/curve_fitting/scaling_params/"
        + site
        + "_scaling_params_"
        + str(remove_artifact)
        + ".csv"
    )
    df = pd.read_csv(path)
    scaling_params_freqs = df["freq"].values
    spread = df["spread"].values
    ratio = df["ratio"].values
    smoothed_spread = df["smoothed_spread"].values
    smoothed_ratio = df["smoothed_ratio"].values

    # multiply by fit scaling...

    # remove outliers
    if remove_outliers:
        lambd_inds = lambd <= 200
        kappa_inds = kappa <= 5
    else:
        lambd_inds = np.full(len(lambd), True)
        kappa_inds = np.full(len(kappa), True)

    scales = [0.5, 0.75, 1.0]
    # for i in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    # for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    for i in scales:
        scaled_log_k = i * np.log(smoothed_ratio[kappa_inds])
        scaled_k = np.exp(scaled_log_k)

        ax[0].plot(scaling_params_freqs[kappa_inds], scaled_k, label=i)
        ax[1].plot(scaling_params_freqs[kappa_inds], scaled_log_k, label=i)

    # plot ratio and kappa
    ax[0].scatter(
        scaling_params_freqs[kappa_inds],
        ratio[kappa_inds],
        c="orange",
        alpha=0.85,
    )
    ax[1].scatter(
        scaling_params_freqs[kappa_inds],
        np.log(ratio[kappa_inds]),
        c="orange",
        alpha=0.85,
    )
    """
    ax[0].plot(
        scaling_params_freqs[kappa_inds],
        smoothed_ratio[kappa_inds],
        c="brown",
    )

    ax[1].plot(
        scaling_params_freqs[kappa_inds],
        np.log(smoothed_ratio[kappa_inds]),
        c="brown",
    )
    """

    ax[0].set_ylabel("ratio")
    ax[0].axhline(y=1.0, c="black")
    ax[0].legend()
    ax[1].axhline(y=0.0, c="black")
    ax[1].set_ylabel("log ratio")
    ax[1].legend()

    ax[1].set_xlabel("Frequency (Hz)")

    plt.show()
