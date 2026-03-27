from dash import Dash, html, dcc, callback, Output, Input, ctx
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import scipy.stats as stats
from scipy import special

from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

import ast

app = Dash()

app.layout = [
    html.H1(children="Curve picking", style={"textAlign": "center"}),
    dcc.Dropdown(["WH01", "WH02", "WH03", "WH04"], "WH01", id="site_selection"),
    dcc.Input(id="n_bins", type="number", min=10, max=500, step=1, value=200),
    # dcc.Upload("Load", id="load_curve"),
    # dcc.Input(id="file_name"),
    # html.Button("Save curve", id="save_curve"),
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
        value=[],
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
            {"label": "plot residuals", "value": True},
        ],
        value=[],
        id="plot_residuals",
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
    dcc.Markdown("Frequency"),
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
    dcc.Markdown("Sigma"),
    dcc.Slider(
        min=0,
        max=2,
        step=0.01,
        value=0.01,
        id="sigma_slider",
    ),
    dcc.Markdown("Lambda"),
    dcc.Slider(
        min=0,
        max=3,
        step=0.01,
        value=1,
        id="lambda_slider",
    ),
    dcc.Markdown("Scale"),
    dcc.Slider(
        min=0,
        max=2,
        step=0.01,
        value=1,
        id="scale_slider",
    ),
]


@app.callback(
    Output(component_id="plt_figure", component_property="src"),
    # Output(component_id="dispersion_curve_figure", component_property="figure"),
    Output(component_id="frequency_dist_figure", component_property="figure"),
    # Input(component_id="dispersion_curve_figure", component_property="figure"),
    Input(component_id="site_selection", component_property="value"),
    Input(component_id="figure_type", component_property="value"),
    Input(component_id="plot_residuals", component_property="value"),
    Input(component_id="transverse_comp", component_property="value"),
    Input(component_id="ln_y_axis", component_property="value"),
    Input(component_id="show_err_dist", component_property="value"),
    Input(component_id="n_bins", component_property="value"),
    Input(component_id="freq_slider", component_property="value"),
    Input(component_id="mu_slider", component_property="value"),
    Input(component_id="sigma_slider", component_property="value"),
    Input(component_id="lambda_slider", component_property="value"),
    Input(component_id="scale_slider", component_property="value"),
)
def update_dispersion_curve(
    # disp_fig,
    site,
    figure_type,
    plot_residuals,
    transverse_comp,
    ln_y_axis,
    show_err_dist,
    n_bins,
    selected_freq,
    mu,
    sigma,
    lambd,
    scale,
):
    """
    Update disperion curve 2D histogram figure.
    Vertical line at selected frequency.
    """
    callback_context = ctx.triggered_id

    plot_residuals = np.sum(plot_residuals) == 1
    transverse_comp = np.sum(transverse_comp) == 1
    ln_y_axis = np.sum(ln_y_axis) == 1
    show_err_dist = np.sum(show_err_dist) == 1

    max_path, curve_path, polygon_path = get_path(
        site, transverse_comp, figure_type, ln_y_axis
    )

    df_max = read_max_file(max_path)
    freqs_grid, vels_grid, freqs, vel_means, vel_meds, stds = compute_dispersion_curve(
        df_max,
    )

    if figure_type == "velocity":
        y_grid = vels_grid
    elif figure_type == "slowness":
        y_grid = 1 / vels_grid

    if ln_y_axis:
        y_grid = np.log(y_grid)

    curve_df = pd.read_csv(curve_path)

    # if callback_context is None:
    with open(polygon_path) as f:
        contents = f.read()
    # polygon = contents.replace("[", "").replace("]", "").split("), (")
    polygon = ast.literal_eval(contents)

    """
    elif callback_context == "dispersion_curve_figure":
        # update dispersion curve selection

        pass

    disp_fig = plotly_hist(
        figure_type,
        freqs,
        freqs_grid,
        y_grid,
        curve_df,
        ln_y_axis,
        selected_freq,
        n_bins,
        polygon,
    )
    """

    pyplot_fig = pyplot_hist(
        figure_type,
        plot_residuals,
        freqs,
        freqs_grid,
        y_grid,
        curve_df,
        selected_freq,
        ln_y_axis,
        n_bins,
        polygon,
    )

    freq_fig = freq_plot(
        figure_type,
        plot_residuals,
        freqs,
        freqs_grid,
        y_grid,
        curve_df,
        selected_freq,
        ln_y_axis,
        show_err_dist,
        n_bins,
        mu,
        sigma,
        lambd,
        scale,
    )

    # return pyplot_fig, disp_fig, freq_fig
    return pyplot_fig, freq_fig


@callback(
    Output(component_id="freq_slider", component_property="min"),
    Output(component_id="freq_slider", component_property="max"),
    Output(component_id="freq_slider", component_property="step"),
    Output(component_id="freq_slider", component_property="marks"),
    Output(component_id="sigma_slider", component_property="min"),
    Output(component_id="sigma_slider", component_property="max"),
    Output(component_id="sigma_slider", component_property="step"),
    Output(component_id="mu_slider", component_property="min"),
    Output(component_id="mu_slider", component_property="max"),
    Output(component_id="mu_slider", component_property="step"),
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
    max_path, curve_path, polygon_path = get_path(
        site, transverse_comp, figure_type, ln_y_axis
    )

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

    if figure_type == "velocity":
        if ln_y_axis:
            min_sigma, max_sigma, step_sigma = 0.1, 2, 0.1
            min_mu, max_mu, step_mu = -2, 2, 0.1
        else:
            min_sigma, max_sigma, step_sigma = 1, 20, 0.5
            min_mu, max_mu, step_mu = -100, 100, 1
    elif figure_type == "slowness":
        if ln_y_axis:
            min_sigma, max_sigma, step_sigma = 0.1, 2, 0.1
            min_mu, max_mu, step_mu = -2, 2, 0.1
        else:
            min_sigma, max_sigma, step_sigma = 0.1, 2, 0.1
            min_mu, max_mu, step_mu = -2, 2, 0.1

    return (
        min_freq,
        max_freq,
        step,
        marks,
        min_sigma,
        max_sigma,
        step_sigma,
        min_mu,
        max_mu,
        step_mu,
    )


def get_path(site, transverse_comp, figure_type, ln_y_axis):
    max_path, curve_path = "", ""

    if site == "WH01":
        if transverse_comp:
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
        if transverse_comp:
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
        if transverse_comp:
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
        if transverse_comp:
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

    polygon_path = curve_path.replace(".csv", ".txt")

    return max_path, curve_path, polygon_path


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


def plotly_hist(
    figure_type,
    freqs,
    freqs_grid,
    y_grid,
    df_curve,
    ln_y_axis,
    selected_freq,
    n_bins,
    polygon,
):
    min_freq = np.log10(np.min(freqs_grid))
    max_freq = np.log10(np.max(freqs_grid))
    step_freq = (max_freq - min_freq) / (len(freqs) + 1)

    min_y = np.min(y_grid)
    max_y = np.max(y_grid)
    step_y = (max_y - min_y) / n_bins

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

    # plot dispersion curve
    y_curve = df_curve["vels"]

    if figure_type == "velocity" and ln_y_axis:
        y_curve = np.log(y_curve)
    elif figure_type == "slowness" and ln_y_axis:
        y_curve = np.exp(y_curve)

    disp_fig.add_trace(
        go.Scatter(
            x=np.log10(df_curve["freqs"]),
            y=y_curve,
            # error_y=dict(type="data", array=new_err, visible=True),
            mode="markers+lines",
        )
    )

    disp_fig.add_vline(x=selected_freq, fillcolor="red", opacity=0.5)

    if callback_context is None:
        polygon_string = "M"
        for ind, (x, y) in enumerate(polygon):
            polygon_string += str(x) + "," + str(y)
            if ind == len(polygon):
                polygon_string += "Z"
            else:
                polygon_string += "L"

        disp_fig.add_selection(path=polygon_string)

    # fig.update_yaxes(range=[50, 2200])
    # fig.update_yaxes(range=[0, 0.0220])

    y_label = figure_type
    if np.sum(ln_y_axis) == 1:
        y_label = "ln(" + y_label + ")"

    disp_fig.update_xaxes(title_text="frequency (Hz)")
    disp_fig.update_yaxes(title_text=y_label)

    return disp_fig


def pyplot_hist(
    figure_type,
    plot_residuals,
    freqs,
    freqs_grid,
    y_grid,
    curve_df,
    selected_freq,
    ln_y_axis,
    n_bins,
    polygon,
):
    # Build the matplotlib figure
    # fig = plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 5))

    freq_bins = np.logspace(
        np.log10(np.min(freqs_grid)), np.log10(np.max(freqs_grid)), len(freqs) + 1
    )
    y_bins = np.linspace(np.min(y_grid), np.max(y_grid), n_bins)

    y_curve = curve_df["vels"]
    if ln_y_axis and figure_type == "velocity":
        y_curve = np.log(y_curve)
    elif not ln_y_axis and figure_type == "slowness":
        y_curve = np.exp(y_curve)

    if plot_residuals:
        # get freqs for dispersion curve
        # get freqs_grid with the same frequencies as the dispersion curve.
        curve_freqs = curve_df["freqs"]

        residuals_freq = []
        residuals_grid = []
        quant_5 = []
        quant_95 = []
        for f in curve_freqs:
            res = list(
                y_grid[np.isclose(freqs_grid, f)].values
                - y_curve[curve_freqs == f].values[0]
            )
            residuals_freq += list(np.repeat(f, len(res)))
            residuals_grid += res
            quant_5.append(np.quantile(res, 0.05))
            quant_95.append(np.quantile(res, 0.95))

        res_bins = np.linspace(np.min(residuals_grid), np.max(residuals_grid), n_bins)
        plt.hist2d(
            residuals_freq,
            residuals_grid,
            bins=[
                freq_bins,
                res_bins,
            ],
            norm=LogNorm(),
        )

        plt.plot(curve_freqs, quant_5, c="black")
        plt.plot(curve_freqs, quant_95, c="black")
    else:
        plt.hist2d(
            freqs_grid,
            y_grid,
            bins=[
                freq_bins,
                y_bins,
            ],
            norm=LogNorm(),
        )

        y_err = None
        if (not ln_y_axis and figure_type == "velocity") or (
            ln_y_axis and figure_type == "slowness"
        ):
            # y_err = curve_df["stds"]
            pass

        plt.errorbar(
            curve_df["freqs"],
            y_curve,
            yerr=y_err,
            marker="o",
            markersize=2,
            c="black",
        )

        """
        ax.add_patch(
            Polygon(
                polygon,
                facecolor="none",
                edgecolor="black",
                linewidth=3,
                # alpha=0.3,
            )
        )
        """
    plt.xscale("log")
    plt.axvline(10**selected_freq, c="red", alpha=0.5)

    y_label = figure_type
    if ln_y_axis:
        y_label = "ln(" + y_label + ")"
    if plot_residuals:
        y_label += " residuals"

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


def freq_plot(
    figure_type,
    plot_residuals,
    freqs,
    freqs_grid,
    y_grid,
    curve_df,
    selected_freq,
    ln_y_axis,
    show_err_dist,
    n_bins,
    mu,
    sigma,
    lambd,
    scale,
):
    min_freq = np.log10(np.min(freqs_grid))
    max_freq = np.log10(np.max(freqs_grid))
    step_freq = (max_freq - min_freq) / (len(freqs) + 1)

    min_y = np.min(y_grid)
    max_y = np.max(y_grid)
    step_y = (max_y - min_y) / n_bins

    y_curve = curve_df["vels"]
    if figure_type == "velocity" and ln_y_axis:
        y_curve = np.log(y_curve)
    elif figure_type == "slowness" and ln_y_axis:
        y_curve = np.exp(y_curve)

    if plot_residuals:
        # get freqs for dispersion curve
        # get freqs_grid with the same frequencies as the dispersion curve.
        curve_freqs = curve_df["freqs"]

        residuals_freq = []
        residuals_grid = []
        for f in curve_freqs:
            res = list(
                y_grid[np.isclose(freqs_grid, f)].values
                - y_curve[curve_freqs == f].values[0]
            )
            residuals_freq += list(np.repeat(f, len(res)))
            residuals_grid += res

        min_res = np.min(residuals_grid)
        max_res = np.max(residuals_grid)
        step_res = (max_res - min_res) / n_bins

        res = np.array(residuals_grid)[np.isclose(residuals_freq, 10**selected_freq)]
        freq_fig = go.Figure(
            data=[
                go.Histogram(
                    x=res,
                    histnorm="probability",
                    # nbinsx=n_bins,
                    xbins=dict(
                        start=min_res, end=max_res, size=step_res
                    ),  # bins used for histogram
                )
            ]
        )
        freq_fig.add_vline(x=np.quantile(res, 0.05), line_color="black", name="5%")
        freq_fig.add_vline(x=np.quantile(res, 0.95), line_color="black", name="95%")
        freq_fig.update_xaxes(range=[min_res, max_res])
    else:
        freq_fig = go.Figure(
            data=[
                go.Histogram(
                    x=y_grid[np.isclose(freqs_grid, 10**selected_freq)],
                    histnorm="probability",
                    # nbinsx=n_bins,
                    xbins=dict(
                        start=min_y, end=max_y, size=step_y
                    ),  # bins used for histogram
                )
            ]
        )
        freq_fig.update_xaxes(range=[min_y, max_y])

    inds = np.isclose(
        curve_df["freqs"], np.repeat(10**selected_freq, len(curve_df["freqs"]))
    )

    x = y_curve[inds].values
    if len(x) == 1:

        if ((not ln_y_axis and figure_type == "velocity") or (
            ln_y_axis and figure_type == "slowness"
        ):
            # err = curve_df["stds"][inds]
            # err = err.values[0]

            # freq_fig.add_vline(x=x[0] - err, fillcolor="black", opacity=0.5, name="std")
            # freq_fig.add_vline(x=x[0] + err, fillcolor="black", opacity=0.5, name="std")

            x = np.linspace(min_res, max_res, 1000)
            pdf = (
                (lambd / 2)
                * np.exp((lambd / 2) * (2 * mu + lambd * sigma**2 - 2 * x))
                * (1 - special.erf((mu + lambd * sigma**2 - x) / (np.sqrt(2) * sigma)))
            )
            if show_err_dist:
                freq_fig.add_trace(
                    go.Scatter(
                        x=x,
                        # y=stats.exponnorm.pdf(x, K, loc=0, scale=sigma),
                        y=scale * pdf,
                        mode="lines",
                        name="EMG",
                    )
                )
            )
        else:
            freq_fig.add_vline(x=x[0], fillcolor="red", opacity=0.5, name="mode")

    y_label = figure_type
    if np.sum(ln_y_axis) == 1:
        y_label = "ln(" + y_label + ")"
    if plot_residuals:
        y_label += " residuals"

    freq_fig.update_xaxes(title_text=y_label)

    return freq_fig


if __name__ == "__main__":
    app.run(debug=True)
