import obspy
from obspy import read, Stream, UTCDateTime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(site):
    # I would recommend applying a bandpass, or even just a high-pass filter >0.1 Hz.

    """
    Extract all vertical channel streams.
    Apply any desired simple processing (demean, detrend, high-pass filter)
    scale/normalize all traces (you might have to use a unique value since some traces have big spikes that would dwarf the normalization)
    extract data as numpy array ( tr.data() ) and absolute times for each trace ( tr.times(type='utcdatetime') )
    Plot all numpy arrays on the same figure. For each subsequent instrument/array, add a constant vertical shift to the array so that they plot above each other (the y axis now becomes meaningless).
    However you managed to shift the arrays along the vertical axis (for example, each is shifted by a value of 1), then you can customize the y axis ticks ( ax.set_yticks([1,2,3,...,N]) ) and the corresponding tick labels ( ax.set_yticklabels( ['TP01', 'TP02', ..., 'TP10' ] ) )
    """

    if site == "WH01" or site == "WH02":
        # data_path = "./data/" + site + "/"
        data_path = "./data/" + site + "_3C_split/"
        coords_path = (
            "./data/" + site + "/txt_files/" + site + "_loc_corrected_geopsy.txt"
        )
    elif site == "WH03" or site == "WH04":
        data_path = "./data/" + site + "/2025_WHY_MTS/"
        coords_path = "./data/" + site + "/" + site + "_loc_corrected_geopsy.txt"

    # read in coords file
    # df = pd.read_csv(coords_path, names=["instrument", "x", "y"], sep="\s+")

    # assemble traces
    traces = []
    for file in os.listdir(data_path):
        if ".miniseed" in file or ".mseed" in file:
            stream = read(data_path + file)
            instrument = stream[0].stats["station"]
            chan = stream[0].stats["channel"]
            if chan == "EPZ" or chan == "HHZ":
                # slice data
                # 6:30 - 9:30
                # 00:48 - 13:17
                tr = stream[0]
                if site == "WH03":
                    tr = tr.trim(
                        # longest section WH03
                        starttime=UTCDateTime(2025, 10, 23, 19, 00, 00),
                        endtime=UTCDateTime(2025, 10, 23, 21, 15, 00),
                    )
                elif site == "WH04":
                    tr = tr.trim(
                        # longest section WH04
                        # starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                        # endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                        # quietest section WH04
                        # starttime=UTCDateTime(2025, 10, 24, 8, 0, 0),
                        # endtime=UTCDateTime(2025, 10, 24, 11, 0, 0),
                        # plot section 1
                        starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                        endtime=UTCDateTime(2025, 10, 24, 6, 30, 0),
                    )

                """
                # set distance
                dist = np.sqrt(
                    df["x"][df["instrument"] == "SS_" + str(instrument)] ** 2
                    + df["y"][df["instrument"] == "SS_" + str(instrument)] ** 2
                )
                tr.stats.distance = dist
                """

                tr.detrend("demean")
                tr.detrend("linear")
                tr.filter("highpass", freq=1.0, corners=2, zerophase=True)

                traces.append(tr)

    n_plots = len(traces)
    if n_plots > 15:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        n_cols = 2
        mid = int(n_plots / 2)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        n_cols = 1
    # fig, axes = plt.subplots(nrows=n_plots, sharex=True, sharey=True)
    station_names = []
    for ind, tr in enumerate(traces):
        print(ind)
        times = pd.date_range(
            np.datetime64(tr.stats["starttime"]),
            np.datetime64(tr.stats["endtime"]),
            periods=tr.stats["npts"],
        ).to_pydatetime()

        tr_data = tr.data
        plot_data = (tr_data - tr_data.min()) / (tr_data.max() - tr_data.min())
        if n_cols == 1:
            ax.plot(times, ind - 0.5 + plot_data, c="black")
        elif n_cols == 2:
            if ind < mid:
                ax[0].plot(times, ind - 0.5 + plot_data, c="black")
            else:
                ax[1].plot(times, (ind - mid) - 0.5 + plot_data, c="black")
        station_names.append(tr.stats["station"])

    if n_cols == 1:
        ax.set_yticks(np.arange(n_plots), station_names)
    elif n_cols == 2:
        ax[0].set_yticks(np.arange(mid), station_names[:mid])
        ax[1].set_yticks(np.arange((n_plots - mid)), station_names[mid:])

    plt.xlabel("time")
    plt.ylabel("instrument")
    if n_cols == 1:
        plt.title("site " + str(site) + " passive seismic recordings")
    else:
        plt.suptitle("site " + str(site) + " passive seismic recordings")
    plt.show()


if __name__ == "__main__":

    plot_raw_data(site="WH04")
