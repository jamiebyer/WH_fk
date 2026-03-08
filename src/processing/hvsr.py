import pathlib

import numpy as np
import matplotlib.pyplot as plt

import os
import hvsrpy

import pandas as pd

# plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)


def run_hvsr(site, file_paths):
    # PREPROCESSING SETTINGS
    """
    detrend: Method for removing trends from the data ("constant", "linear", etc.)
    window_length_in_seconds: Length of time windows for analysis
    orient_to_degrees_from_north: Orientation correction for horizontal components
    filter_corner_frequencies_in_hz: Low and high cutoff frequencies for bandpass filtering
    """
    preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
    preprocessing_settings.detrend = "linear"
    preprocessing_settings.window_length_in_seconds = 100
    preprocessing_settings.orient_to_degrees_from_north = 0.0
    preprocessing_settings.filter_corner_frequencies_in_hz = (None, None)
    preprocessing_settings.ignore_dissimilar_time_step_warning = False

    # PROCESSING SETTINGS
    """
    window_type_and_width: Tapering window type and width for spectral analysis
    smoothing: Spectral smoothing parameters (operator, bandwidth, frequencies)
    method_to_combine_horizontals: Method for combining horizontal components ("geometric_mean", "arithmetic_mean", etc.)
    """
    processing_settings = hvsrpy.settings.HvsrTraditionalProcessingSettings()
    processing_settings.window_type_and_width = ("tukey", 0.2)
    processing_settings.smoothing = dict(
        operator="konno_and_ohmachi",
        bandwidth=40,
        center_frequencies_in_hz=np.geomspace(0.2, 50, 200),
    )
    processing_settings.method_to_combine_horizontals = "geometric_mean"
    processing_settings.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

    # RUN HVSR

    srecords = hvsrpy.read([file_paths])
    srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
    """
    srecords = hvsrpy.sta_lta_window_rejection(
        srecords,
        sta_seconds=1,
        lta_seconds=30,
        min_sta_lta_ratio=0.2,
        max_sta_lta_ratio=2.5,
        components=("ns", "ew", "vt"),
    )
    """
    hvsr = hvsrpy.process(srecords, processing_settings)

    """
    The hvsrpy library provides several window rejection methods:
    Frequency Domain Rejection: Rejects windows based on their deviation from the median curve
    STA/LTA Rejection: Rejects windows based on short-term to long-term average ratio
    Maximum Value Rejection: Rejects windows that exceed a maximum amplitude threshold
    Manual Rejection: Allows manual selection of windows to reject
    """
    hvsrpy.frequency_domain_window_rejection(hvsr)

    # peak_frequency = hvsr.peak_frequency
    # peak_amplitude = hvsr.peak_amplitude

    # Check HVSR results against SESAME criteria
    # sesame_results = hvsrpy.sesame.check_criteria(hvsr)

    # RESULTS
    (fig, ax) = hvsrpy.plot_single_panel_hvsr_curves(
        hvsr,
    )
    ax.get_legend().remove()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.show()

    # SAVE RESULTS

    save_figure = False
    save_results = True
    out_path = (
        "./results/hvsr/" + site + "/" + file_paths[0].split("/")[-1].split(".mseed")[0]
    )

    if save_figure:
        fname = f"{out_path.replace("results", "figures")}.png"
        fig.savefig(fname)
        plt.close()
        print(f"Figure saved successfully to {fname}!")

    if save_results:
        fname = f"{out_path}.csv"
        hvsrpy.object_io.write_hvsr_object_to_file(hvsr, fname)
        print(f"Results saved successfully to {fname}!")


def plot_full_site():
    """
    csv columns:
    frequency (Hz),hvsr curve 1,hvsr curve 2,...,hvsr curve n,mean curve (lognormal),mean curve std (lognormal)
    """
    site = "WH02"

    dir_path = "./results/hvsr/" + site + "/"

    fig, ax = plt.subplots(figsize=(15, 7))
    count_rows = True
    n = 0
    means = []
    for f in os.listdir(dir_path):
        # """
        if site == "WH03" and n == 0:
            n += 1
            continue
        # """
        # for just first loop, find number of rows to skip
        if count_rows:
            with open(dir_path + f) as first_f:
                lines = first_f.readlines()
            n_rows = 1
            for line in lines:
                if line.startswith("#"):
                    n_rows += 1
                    if line.startswith("# frequency (Hz)"):
                        cols = line.replace("\n", "").replace("# ", "").split(",")

            count_rows = False

        if (
            ((site == "WH01" or site == "WH02") and f.startswith("T"))
            or (
                site == "WH03"
                and (f.startswith("453025242") or f.startswith("453025057"))
            )
            or site == "WH04"
            and (f.startswith("453024625") or f.startswith("453025257"))
        ):
            color = "red"
        else:
            color = "black"

        df = pd.read_csv(dir_path + f, skiprows=n_rows, names=cols)
        plt.errorbar(
            df["frequency (Hz)"],
            df["mean curve (lognormal)"],
            df["mean curve std (lognormal)"],
            c=color,
            alpha=0.5,
            # label=f,
        )
        # plt.legend()
        means.append(df["mean curve (lognormal)"])
        # print(df["mean curve (lognormal)"].shape)

    # print(np.array(means).shape)
    # plt.errorbar(df["frequency (Hz)"], np.mean(means), np.std(means), c="red")

    plt.xscale("log")
    plt.title(site + ", n instruments: " + str(len(means)))
    plt.show()


def plot_peak_freqs():
    # import mapping?
    # plot array configuration with peak frequencies
    pass


def run_hvsr_files():
    site = "WH04"

    if site == "WH01":
        dir_path = "./data/WH01_3C_split/"
        # "0252_WH01.mseed"
    elif site == "WH02":
        dir_path = "./data/WH02_3C_split/"
        # "TP02_WH02.mseed"
        # "TP07_WH02.mseed"
    elif site == "WH03":
        dir_path = "./data/WH03/mseed_files/"
    elif site == "WH04":
        # dir_path = "./data/WH04/longest_slice/"
        dir_path = "./data/WH04/quietest_slice/"

    for f in os.listdir(dir_path):
        if (".mseed" not in f and ".miniseed" not in f) or "E." not in f:
            continue
        if site == "WH01" and "0252_WH01" in f:
            continue
        if site == "WH02" and ("TP02_WH02" in f or "TP07_WH02" in f):
            continue
        # print(dir_path + f)
        paths = [
            dir_path + f,
            (dir_path + f).replace("E.", "N."),
            (dir_path + f).replace("E.", "Z."),
        ]
        print(paths)
        run_hvsr(
            site,
            paths,
        )


if __name__ == "__main__":
    plot_full_site()
