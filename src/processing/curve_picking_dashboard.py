from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import scipy.stats as stats

from matplotlib.colors import LogNorm

app = Dash()

app.layout = [
    html.H1(children="Curve picking", style={"textAlign": "center"}),
    dcc.Dropdown(["WH01", "WH02", "WH03", "WH04"], "WH01", id="site_selection"),
    dcc.Input(id="n_bins", type="number", min=10, max=500, step=1, value=100),
    dcc.Upload("Load", id="load_curve"),
    dcc.Input(id="file_name"),
    html.Button("Save curve", id="save_curve"),
    dcc.Dropdown(["velocity", "slowness"], "velocity", id="figure_type"),
    dcc.Checklist(
        options=[
            {"label": "transverse", "value": True},
        ],
        value=[],
        id="transverse_comp",
    ),
    dcc.Checklist(
        options=[
            {"label": "ln y-axis", "value": True},
        ],
        value=[True],
        id="ln_y_axis",
    ),
    dcc.Checklist(
        options=[
            {"label": "ln counts", "value": True},
        ],
        value=[True],
        id="ln_counts",
    ),
    dcc.Checklist(
        options=[
            {"label": "show curve", "value": True},
        ],
        value=[True],
        id="show_curve",
    ),
    dcc.Checklist(
        options=[
            {"label": "show error distribution", "value": True},
        ],
        value=[True],
        id="show_err_dist",
    ),
    html.Img(id="plt_figure"),
    # dcc.Graph(id="dispersion_curve_figure"),
    dcc.Slider(
        min=0,
        max=10,
        step=None,
        marks={0: "0"},
        value=0,
        # tooltip={"placement": "bottom"},
        id="freq_slider",
    ),
    dcc.Graph(id="frequency_dist_figure"),
]


@app.callback(
    Output(component_id="plt_figure", component_property="src"),
    Input(component_id="site_selection", component_property="value"),
    Input(component_id="figure_type", component_property="value"),
    Input(component_id="transverse_comp", component_property="value"),
    Input(component_id="ln_y_axis", component_property="value"),
    Input(component_id="n_bins", component_property="value"),
    Input(component_id="freq_slider", component_property="value"),
)
def update_dispersion_curve(
    site, figure_type, transverse_comp, ln_y_axis, n_bins, freq
):
    # Build the matplotlib figure
    fig = plt.figure(figsize=(14, 5))

    max_path, curve_path = get_path(site, transverse_comp, figure_type, ln_y_axis)

    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    if figure_type == "velocity":
        y_grid = vels_grid
    elif figure_type == "slowness":
        y_grid = 1 / vels_grid

    if np.sum(ln_y_axis) == 1:
        y_grid = np.log(y_grid)

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    y_bins = np.linspace(np.min(y_grid), np.max(y_grid), n_bins)

    plt.hist2d(
        freqs_grid,
        y_grid,
        bins=[
            freq_bins,
            y_bins,
        ],
        norm=LogNorm(),
    )

    df = pd.read_csv(curve_path)
    y_curve = df["vels"]

    # if figure_type == "velocity":
    #     y_curve = df["vels"]
    # elif figure_type == "slowness":
    #     y_curve = 1 / df["vels"]

    if np.sum(ln_y_axis) == 1 and figure_type == "velocity":
        y_curve = np.log(y_curve)
    elif np.sum(ln_y_axis) == 0 and figure_type == "slowness":
        y_curve = np.exp(y_curve)

    y_err = None
    if (np.sum(ln_y_axis) == 0 and figure_type == "velocity") or (
        np.sum(ln_y_axis) == 1 and figure_type == "slowness"
    ):
        y_err = df["stds"]

    plt.errorbar(
        df["freqs"],
        y_curve,
        yerr=y_err,
        marker="o",
        markersize=2,
        c="black",
    )

    plt.xscale("log")
    plt.axvline(10**freq, c="red", alpha=0.5)

    y_label = figure_type
    if np.sum(ln_y_axis) == 1:
        y_label = "ln(" + y_label + ")"

    plt.xlabel("frequency (Hz)")
    plt.ylabel(y_label)

    plt.colorbar(label="counts")

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f"data:image/png;base64,{fig_data}"

    return fig_bar_matplotlib


# """


@app.callback(
    # Output(component_id="dispersion_curve_figure", component_property="figure"),
    Output(component_id="frequency_dist_figure", component_property="figure"),
    Input(component_id="site_selection", component_property="value"),
    Input(component_id="figure_type", component_property="value"),
    Input(component_id="transverse_comp", component_property="value"),
    Input(component_id="ln_y_axis", component_property="value"),
    Input(component_id="n_bins", component_property="value"),
    Input(component_id="freq_slider", component_property="value"),
)
def update_dispersion_curve(
    site, figure_type, transverse_comp, ln_y_axis, n_bins, freq
):
    """
    Update disperion curve 2D histogram figure.
    Vertical line at selected frequency.
    """

    max_path, curve_path = get_path(site, transverse_comp, figure_type, ln_y_axis)

    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    if figure_type == "velocity":
        y_grid = vels_grid
    elif figure_type == "slowness":
        y_grid = 1 / vels_grid

    if np.sum(ln_y_axis) == 1:
        y_grid = np.log(y_grid)

    min_freq = np.log10(np.min(freqs_grid))
    max_freq = np.log10(np.max(freqs_grid))
    step_freq = (max_freq - min_freq) / (len(freqs) + 1)

    min_y = np.min(y_grid)
    max_y = np.max(y_grid)
    step_y = (max_y - min_y) / n_bins

    """
    # plot frequency and velocity 2D histogram
    disp_fig = go.Figure(
        go.Histogram2d(
            x=np.log10(freqs_grid),
            y=y_grid,
            # z=h,
            # histnorm="probability",
            # autobinx=False,
            # xbins=dict(start=min_freq, end=max_freq, size=step_freq),
            # autobiny=False,
            # ybins=dict(start=min_y, end=max_y, size=step_y),
            nbinsx=len(freqs) + 1,
            nbinsy=n_bins,
            # zmin=5,
            # zmax=0.00000001,
            # colorscale="ylorrd",
        )
    )
    """
    # plot dispersion curve
    df = pd.read_csv(curve_path)
    y_curve = df["vels"]

    if figure_type == "velocity" and np.sum(ln_y_axis) == 1:
        y_curve = np.log(y_curve)
    elif figure_type == "slowness" and np.sum(ln_y_axis) == 0:
        y_curve = np.exp(y_curve)

    """
    disp_fig.add_trace(
        go.Scatter(
            x=np.log10(df["freqs"]),
            y=y_curve,
            # error_y=dict(type="data", array=new_err, visible=True),
            mode="markers+lines",
        )
    )

    disp_fig.add_vline(x=freq, fillcolor="red", opacity=0.5)
    """
    # fig.update_yaxes(range=[50, 2200])
    # fig.update_yaxes(range=[0, 0.0220])

    y_label = figure_type
    if np.sum(ln_y_axis) == 1:
        y_label = "ln(" + y_label + ")"

    # disp_fig.update_xaxes(title_text="frequency (Hz)")
    # disp_fig.update_yaxes(title_text=y_label)

    freq_fig = go.Figure(
        data=[
            go.Histogram(
                x=y_grid[np.isclose(freqs_grid, 10**freq)],
                # histnorm="probability",
                # nbinsx=n_bins,
                xbins=dict(
                    start=min_y, end=max_y, size=step_y
                ),  # bins used for histogram
            )
        ]
    )

    inds = np.isclose(df["freqs"], np.repeat(10**freq, len(df["freqs"])))
    x = y_curve[inds].values
    if len(x) == 1:
        freq_fig.add_vline(
            x=x[0],
            fillcolor="red",
            opacity=0.5,
        )

        if (np.sum(ln_y_axis) == 0 and figure_type == "velocity") or (
            np.sum(ln_y_axis) == 1 and figure_type == "slowness"
        ):
            err = df["stds"][inds]
            err = err.values[0]

            freq_fig.add_vline(
                x=x[0] - err,
                fillcolor="black",
                opacity=0.5,
            )
            freq_fig.add_vline(
                x=x[0] + err,
                fillcolor="black",
                opacity=0.5,
            )

            """
            mu = x[0]
            # variance = err
            sigma = err
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            freq_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=stats.norm.pdf(x, mu, sigma),
                    mode="lines",
                )
            )
            """
    freq_fig.update_xaxes(range=[min_y, max_y])
    freq_fig.update_xaxes(title_text=y_label)

    return freq_fig


@callback(
    Output(component_id="freq_slider", component_property="min"),
    Output(component_id="freq_slider", component_property="max"),
    Output(component_id="freq_slider", component_property="step"),
    Output(component_id="freq_slider", component_property="marks"),
    Input(component_id="site_selection", component_property="value"),
    Input(component_id="transverse_comp", component_property="value"),
    Input(component_id="figure_type", component_property="value"),
    Input(component_id="ln_y_axis", component_property="value"),
)
def update_frequency_slider(site, transverse_comp, figure_type, ln_y_axis):
    """
    Get frequency options from site selection.

    (If plotting dispersion curve, only show frequencies from dispersion curve.
     Otherwise, plot all frequencies)
    """
    max_path, curve_path = get_path(site, transverse_comp, figure_type, ln_y_axis)

    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    # freq_bins = np.logspace(
    #    np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    # )

    df = pd.read_csv(curve_path)

    marks = None
    """
    marks = {}
    for ind, f in enumerate(df["freqs"]):
        if (ind % 4) == 0:
            marks[np.log10(f)] = str(np.round(f, 2))
    """
    min_freq = np.log10(np.min(df["freqs"]))
    max_freq = np.log10(np.max(df["freqs"]))

    step = (max_freq - min_freq) / (len(df["freqs"]) - 1)

    return min_freq, max_freq, step, marks


def get_path(site, transverse_comp, figure_type, ln_y_axis):
    if site == "WH01":
        if np.sum(transverse_comp) == 1:
            max_path = "./results/fk/final/conventionaltransverse-WH01-default04.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH01-2C.csv"
            elif figure_type == "slowness":
                # if np.sum(ln_y_axis) == 1:
                curve_path = "./results/curves/curve-WH01-transverse-slowness-True.csv"
        else:
            max_path = "./results/fk/final/conventional-WH01_3C_split-default08.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH01-1C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH01-vertical-slowness-True.csv"
    elif site == "WH02":
        if np.sum(transverse_comp) == 1:
            max_path = "./results/fk/final/conventionaltransverse-WH02-default04.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH02-2C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH02-transverse-slowness-True.csv"
        else:
            max_path = "./results/fk/final/conventional-WH02_3C_split-default08.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH02-1C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH02-vertical-slowness-True.csv"
    elif site == "WH03":
        if np.sum(transverse_comp) == 1:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH03-sliced-default04.max"
            )
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH03-2C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH03-transverse-slowness-True.csv"
        else:
            max_path = "./results/fk/final/conventional-WH03-default08.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH03-1C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH03-vertical-slowness-True.csv"
    elif site == "WH04":
        if np.sum(transverse_comp) == 1:
            max_path = (
                "./results/fk/final/conventionaltransverse-WH04-longest-default04.max"
            )
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH04-2C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH04-transverse-slowness-True.csv"
        else:
            max_path = "./results/fk/final/conventional-WH04-longest-default08.max"
            if figure_type == "velocity":
                curve_path = "./results/curves/curve-WH04-1C.csv"
            elif figure_type == "slowness":
                curve_path = "./results/curves/curve-WH04-vertical-slowness-True.csv"

    return max_path, curve_path


def read_max_file(max_file):
    """
    Plot geopsy ".max" file.
    Gives time, frequency, slowness, azimuth, power, --
    """
    # read max file
    # determine how many lines to skip when reading pd dataframe
    with open(max_file, "r") as file:
        # Read the first line
        line = file.readline()
        ind = 0
        while line:
            if "# BEGIN DATA" in line:
                ind += 3
                break
            line = file.readline()  # Read the next line
            ind += 1

    # column names (slightly different for old max files)
    names = [
        "abs_time",
        "frequency",
        # "polarization",
        "slowness",
        "azimuth",
        "",
        "el",
        "no",
        "power",
        "valid",
    ]

    # read ".max" file as dataframe
    # df = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)
    df = pd.read_csv(max_file, skiprows=ind, sep="\\s", names=names)
    return df


def compute_dispersion_curve(df, err_thresh=None, freq_outliers=[], vel_outliers=[]):
    """
    compute dispersion curve from .max file df.
    get median and std for each frequency.
    minimum threshold for error.
    """
    freqs_grid = df["frequency"]
    vels_grid = 1 / df["slowness"]
    # az = df["azimuth"]
    # power = df["power"]

    # compute dispersion curve
    freqs_curve = np.unique(freqs_grid)
    vel_meds_curve = []
    vel_means_curve = []
    stds_curve = []
    # for each frequency, save the median velocity, and
    # compute the standard deviation
    for f in freqs_curve:
        vels = vels_grid[freqs_grid == f]
        if len(freq_outliers) > 0:
            ind = np.argmin(np.abs(freq_outliers - f))

            if np.abs(freq_outliers[ind] - f) < 0.01:
                vels = vels[vels < vel_outliers[ind]]

        vel_med = np.median(vels)
        vel_mean = np.mean(vels)
        std = np.std(vels)
        vel_meds_curve.append(vel_med)
        vel_means_curve.append(vel_mean)
        stds_curve.append(std)

    # error threshold
    if err_thresh is not None:
        ind = np.argmin(np.abs(freqs_curve - err_thresh))
        stds_curve[ind:] = (len(stds_curve) - ind) * [stds_curve[ind]]

    return (
        freqs_grid,
        vels_grid,
        freqs_curve,
        np.array(vel_meds_curve),
        np.array(vel_means_curve),
        np.array(stds_curve),
    )


if __name__ == "__main__":
    app.run(debug=True)
