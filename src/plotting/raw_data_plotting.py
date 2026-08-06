import obspy
from obspy import read, Stream, UTCDateTime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


def plot_raw_data(site):
    """
    Extract all vertical channel streams.
    Apply any desired simple processing (demean, detrend, high-pass filter)
    scale/normalize all traces (you might have to use a unique value since some traces have big spikes that would dwarf the normalization)
    """

    if site == "WH01" or site == "WH02":
        # data_path = "./data/" + site + "/"
        data_path = "./data/" + site + "_3C_split/"
    elif site == "WH03" or site == "WH04":
        data_path = "./data/" + site + "/2025_WHY_MTS/"

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
                        starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                        endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                        # quietest section WH04
                        # starttime=UTCDateTime(2025, 10, 24, 8, 0, 0),
                        # endtime=UTCDateTime(2025, 10, 24, 11, 0, 0),
                        # plot section 1
                        # starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                        # endtime=UTCDateTime(2025, 10, 24, 6, 30, 0),
                    )

                tr.detrend("demean")
                tr.detrend("linear")
                tr.filter("highpass", freq=1.0, corners=2, zerophase=True)

                if site == "WH04":
                    tr.decimate(factor=4, strict_length=False)

                traces.append(tr)

    n_plots = len(traces)
    if n_plots > 15:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        n_cols = 2
        mid = int(n_plots / 2)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
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

    myFmt = mdates.DateFormatter("%H:%M")
    if n_cols == 1:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_yticks(np.arange(n_plots), station_names)
        ax.xaxis.set_major_formatter(myFmt)
    elif n_cols == 2:
        ax[0].set_yticks(np.arange(mid), station_names[:mid])
        ax[1].set_yticks(np.arange((n_plots - mid)), station_names[mid:])
        ax[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[0].xaxis.set_major_formatter(myFmt)
        ax[1].xaxis.set_major_formatter(myFmt)

    if n_cols == 1:
        plt.xlabel("time")
        plt.ylabel("instrument")
    elif n_cols == 2:
        fig.text(0.5, 0.04, "time", ha="center")
        ax[0].set_ylabel("instrument")

    plt.savefig("./figures/raw_data/" + site + "_traces.png")


def plot_power_spectrum(site):
    # read in miniseeds for full site...
    if site == "WH01" or site == "WH02":
        # data_path = "./data/" + site + "/"
        data_path = "./data/" + site + "_3C_split/"
    elif site == "WH03" or site == "WH04":
        data_path = "./data/" + site + "/2025_WHY_MTS/"

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
                fs = tr.stats["delta"]

                if site == "WH03":
                    tr = tr.trim(
                        # longest section WH03
                        starttime=UTCDateTime(2025, 10, 23, 19, 00, 00),
                        endtime=UTCDateTime(2025, 10, 23, 21, 15, 00),
                    )
                elif site == "WH04":
                    tr = tr.trim(
                        # longest section WH04
                        starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                        endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                    )

                tr.detrend("demean")
                tr.detrend("linear")
                tr.filter("highpass", freq=1.0, corners=2, zerophase=True)

                if site == "WH04":
                    tr.decimate(factor=4, strict_length=False)

                traces.append(tr)

    # Applying FFT
    freqs = []
    ffts = []
    for tr in traces:
        # freq = np.fft.fftfreq(len(tr), d=1 / fs)
        freq = np.fft.fftfreq(len(tr), d=fs)
        fft_result = np.fft.fft(tr.data)

        freqs += list(freq)
        ffts += list(np.real(fft_result))

        # Plotting the spectrum
        # plt.plot(freq, np.abs(fft_result), c="grey", alpha=0.1)

    plt.hist2d(freqs, ffts, bins=[200, 200], cmin=1)
    plt.colorbar()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(site)
    plt.show()


if __name__ == "__main__":
    # for site in ["WH01", "WH02", "WH03", "WH04"]:
    plot_raw_data(site="WH04")
