import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Polygon as plt_polygon
from matplotlib.collections import PatchCollection

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as shp_polygon

from scipy.stats import mode
import pandas as pd

import sys

sys.path.append("./src/")
sys.path.append("../mcmc/src/")
from processing.dispersion_curves import compute_dispersion_curve, read_max_file


class CurvePicking:
    """
    From: https://en.matplotlib.net/stable/gallery/widgets/polygon_selector_demo.html

    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self):
        ax, hist, freqs, vels = self.plot_figure()
        self.ax = ax
        self.hist_counts, self.x_bins, self.y_bins = hist[0], hist[1], hist[2]

        # scatter plot for dispersion curve
        self.scatter = self.ax.scatter([], [], c="black", s=3)
        self.errorbar = self.ax.errorbar([], [], yerr=[], c="black")
        self.plot_freqs, self.plot_vels, self.plot_stds = [], [], []

        # use np where to find inds of histogram
        # get max from hist counts

        self.canvas = ax.figure.canvas
        # self.collection = collection

        self.freqs = freqs.values
        self.vels = vels.values
        self.pts = [Point(freqs[i], vels[i]) for i in range(len(freqs))]
        # create 2d array of polygons for the histogram bins
        self.bins = np.array(
            [
                [
                    shp_polygon(
                        (
                            (self.x_bins[i], self.y_bins[j]),
                            (self.x_bins[i + 1], self.y_bins[j]),
                            (self.x_bins[i + 1], self.y_bins[j + 1]),
                            (self.x_bins[i], self.y_bins[j + 1]),
                        )
                    )
                    for i in range(len(self.x_bins) - 1)
                ]
                for j in range(len(self.y_bins) - 1)
            ]
        )

        # for indicating selected data bins
        """
        patches = []
        for b in self.bins.flatten():
            coords = b.exterior.coords.xy
            coords_x = list(coords[0])
            coords_y = list(coords[1])
            bin_coords = [[coords_x[i], coords_y[i]] for i in range(len(coords_x))]

            polygon = plt_polygon(bin_coords)
            patches.append(polygon)

        self.patches = PatchCollection(patches, alpha=0.5, c="white")
        ax.add_collection(self.patches)
        """

        # Polygon selection
        self.poly = PolygonSelector(ax, self.onselect)  # , draw_bounding_box=True)
        self.verts = []

    def onselect(self, verts):
        self.verts = verts
        # get bins that are contained completely in the polygon
        polygon = shp_polygon(verts)
        selected_inds = np.array(
            [
                [polygon.contains(self.bins[i][j]) for j in range(len(self.bins[i]))]
                for i in range(len(self.bins))
            ]
        )

        # should be able to get the exact val for freq and velocity
        # check for if there's no max / multiple
        """
        curve = []
        for row_ind, selected_row in enumerate(selected_inds.T):
            if np.sum(selected_row > 0):

                max_ind = np.argmax(self.hist_counts[row_ind][selected_row])

                f = (self.x_bins[row_ind] + self.x_bins[row_ind + 1]) / 2
                v = (
                    self.y_bins[:-1][selected_row][max_ind]
                    + self.y_bins[1:][selected_row][max_ind]
                ) / 2
                curve.append([f, v])
        """

        curve = []
        new_f, new_v, new_e = [], [], []
        for row_ind, selected_row in enumerate(selected_inds.T):
            # for each unique freq in selected polygon
            # get freq range for bin
            # make sure only one unique freq from grid vals
            # get mode of vels_grid within freq range
            # get std of vels_grid within freq range
            if np.sum(selected_row > 0):
                freq_inds = (self.freqs >= self.x_bins[row_ind]) & (
                    self.freqs <= self.x_bins[row_ind + 1]
                )
                f = np.unique(self.freqs[freq_inds])[0]
                if len(np.unique(self.freqs[freq_inds])) != 1:
                    print("multiple freqs")
                    break

                # get largest vel hist bin
                # average vel values within the bin
                max_ind = np.argmax(self.hist_counts[row_ind][selected_row])

                vel_inds = (self.vels >= self.y_bins[:-1][selected_row][max_ind]) & (
                    self.vels <= self.y_bins[1:][selected_row][max_ind]
                )
                v = np.mean(self.vels[vel_inds])

                err_inds = (self.vels >= self.y_bins[:-1][selected_row][0]) & (
                    self.vels <= self.y_bins[1:][selected_row][-1]
                )
                e = np.std(self.vels[err_inds])

                curve.append([f, v])

                new_f.append(f)
                new_v.append(v)
                new_e.append(e)

        # update figure data selection opacity / outline data

        self.scatter.set_offsets(curve)
        self.adjustErrbarxy(
            self.errorbar, np.array(new_f), np.array(new_v), np.array(new_e)
        )

        self.plot_freqs, self.plot_vels, self.plot_stds = (
            np.array(new_f),
            np.array(new_v),
            np.array(new_e),
        )

        # set the opacity of bins/polygons which are not fully contained within the polyon selector

        # self.patches.set_visible(selected_inds.flatten())

        self.canvas.draw_idle()

    def disconnect(self):
        # save df of dispersion curve to file
        df = pd.DataFrame(
            {
                "freqs": self.plot_freqs,
                "vels": self.plot_vels,
                "stds": self.plot_stds,
            }
        )
        df.to_csv("./results/curves/curve-WH04-2C.csv")

        with open("./results/curves/curve-WH04-2C.txt", "w") as f:
            f.write(str(self.verts))

        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def plot_figure(self):
        # 1C
        # max_path = "./results/fk/final/conventional-WH01_3C_split-default08.max"
        # max_path = "./results/fk/final/conventional-WH02_3C_split-default08.max"
        # max_path = "./results/fk/final/conventional-WH03-default08.max"
        # max_path = "./results/fk/final/conventional-WH04-longest-default08.max"
        # 2C
        # max_path = "./results/fk/final/conventionaltransverse-WH01-default04.max"
        # max_path = "./results/fk/final/conventionaltransverse-WH02-default04.max"
        # max_path = "./results/fk/final/conventionaltransverse-WH03-sliced-default04.max"
        max_path = (
            "./results/fk/final/conventionaltransverse-WH04-longest-default04.max"
        )

        df_max = read_max_file(max_path)
        freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = (
            compute_dispersion_curve(
                df_max,
            )
        )

        freq_bins = np.logspace(
            np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
        )
        vel_bins = np.logspace(
            np.log10(np.min(vels_grid)), np.log10(np.max(vels_grid)), len(vel_meds) + 1
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        # plot frequency and velocity 2D histogram
        hist = plt.hist2d(
            freqs_grid,
            vels_grid,
            bins=[
                freq_bins,
                vel_bins,
            ],
            cmap="coolwarm",
            norm=LogNorm(),
        )

        plt.xscale("log")
        plt.yscale("log")
        # plt.ylim([100, 2200])
        plt.ylim([50, 2200])

        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase velocity (m/s)")

        plt.colorbar(label="counts")

        # plot dispersion curve with errors
        # plt.plot(freqs_curve, vels_curve)

        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)

        plt.grid(True)

        plt.tight_layout()

        # path = "./figures/WH02/1C/conventional-WH02-default08.png"
        # path = "./figures/WH04/2C/conventionaltransverse-WH04-longest-default03.png"
        # path = "./figures/WH01/3C/rtbf-WH01-test01.png"
        # plt.savefig(path)
        # plt.show()

        return ax, hist, freqs_grid, vels_grid

    def adjustErrbarxy(self, errobj, x, y, y_error):

        # ln, (errx_top, errx_bot, erry_top, erry_bot), (barsx, barsy) = errobj
        ln, caplines, (barsy) = errobj
        barsy = barsy[0]

        x_base = x
        y_base = y

        # xerr_top = x_base + x_error
        # xerr_bot = x_base - x_error
        yerr_top = y_base + y_error
        yerr_bot = y_base - y_error

        # errx_top.set_xdata(xerr_top)
        # errx_bot.set_xdata(xerr_bot)
        # errx_top.set_ydata(y_base)
        # errx_bot.set_ydata(y_base)

        # erry_top.set_xdata(x_base)
        # erry_bot.set_xdata(x_base)
        # erry_top.set_ydata(yerr_top)
        # erry_bot.set_ydata(yerr_bot)

        # new_segments_x = [np.array([[xt, y], [xb,y]]) for xt, xb, y in zip(xerr_top, xerr_bot, y_base)]
        new_segments_y = [
            np.array([[x, yt], [x, yb]])
            for x, yt, yb in zip(x_base, yerr_top, yerr_bot)
        ]
        # barsx.set_segments(new_segments_x)
        barsy.set_segments(new_segments_y)


if __name__ == "__main__":
    selector = CurvePicking()

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected polygon
    print("\nSelected polygon:")
    print(selector.verts)
