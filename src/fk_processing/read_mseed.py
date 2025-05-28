import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import os


def read_data():
    in_path = "./data/WH02/mseeds"
    for f in os.listdir(in_path):
        if not f.endswith(".mseed"):
            continue
        st = read(in_path + "/" + f)
        print(st)
