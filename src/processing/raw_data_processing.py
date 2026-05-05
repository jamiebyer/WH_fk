import matplotlib.pyplot as plt
import numpy as np


def combine_mseed_files():
    # read from coordinate file
    data_path = "./data/WH03/2025_WHY_MTS/"
    instrument_names = [
        "C",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "XE1",
        "XE2",
        "XE3",
        "XE4",
        "XW1",
        "XW2",
        "XW3",
        "XW4",
    ]

    for inst in instrument_names:
        for coord in ["N", "E", "Z"]:
            traces = []
            for file in os.listdir(data_path):
                if inst not in file or "." + coord + "." not in file:
                    continue
                stream = read(data_path + file)
                traces.append(stream[0])
            output_fname = "./data/WH03/mseed_files/" + inst + coord + ".miniseed"
            combined = Stream(traces)
            combined.write(output_fname, format="MSEED")


def split_miniseed_components():
    sites = [
        "0240",
        "0252",
        "0253",
        "0424",
        "0526",
        "TP01",
        "TP02",
        "TP03",
        "TP04",
        "TP05",
        "TP06",
        "TP07",
        "TP09",
        "TP10",
    ]

    file_names = ["./data/WH02_3C/" + s + "_WH02.mseed" for s in sites]

    for f in file_names:
        st = read(f)
        for s in st:
            chan = s.stats["channel"][-1]
            s.write(
                f.replace("WH02_3C", "WH02_3C_split").replace(
                    ".mseed", "" + chan + ".mseed"
                ),
                format="MSEED",
            )


def slice_noise_data():
    data_path = "./data/WH03/2025_WHY_MTS/"
    # data_path = "./data/WH04/2025_WHY_MTS/"
    for file in os.listdir(data_path):
        if ".miniseed" in file:
            stream = read(data_path + file)
            # 6:30 - 9:30
            # 00:48 - 13:17
            tr = stream[0]
            tr = tr.trim(
                # longest section WH03
                starttime=UTCDateTime(2025, 10, 23, 19, 00, 00),
                endtime=UTCDateTime(2025, 10, 23, 21, 15, 00),
                # longest section WH04
                # starttime=UTCDateTime(2025, 10, 24, 1, 45, 0),
                # endtime=UTCDateTime(2025, 10, 24, 12, 30, 0),
                # quietest section WH04
                # starttime=UTCDateTime(2025, 10, 24, 8, 0, 0),
                # endtime=UTCDateTime(2025, 10, 24, 11, 0, 0),
            )

            output_fname = "./data/WH03/mseed_files/" + file
            sliced_tr = Stream(tr)
            sliced_tr.write(output_fname, format="MSEED")


def ambient_noise_data(site):
    stations = [
        "0240",
        "0252",
        "0253",
        "0424",
        "0526",
        "TP01",
        "TP02",
        "TP03",
        "TP04",
        "TP05",
        "TP06",
        "TP07",
        "TP09",
        "TP10",
    ]

    dir_path = "./data/" + site + "/mseeds/"
    for s in stations:
        st = read(dir_path + s + "_" + site + ".mseed")
        print(st[0].stats)
