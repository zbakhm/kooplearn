import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash_bootstrap_components as dbc
from dash import dcc, html

import base64, io
import pandas as pd


PLOTLY_THEME = "plotly_dark"


def create_plot_eigs(infos, min_freq=None, max_freq=None):
    # infos is a dataframe with a 'eigval_real' and 'eigval_imag' columns
    if min_freq is None:
        min_freq = infos["frequency"].min()
    if max_freq is None:
        max_freq = infos["frequency"].max()
    fig = go.Figure(
        go.Scatter(
            x=infos["eigval real"],
            y=infos["eigval imag"],
            mode="markers",
            marker_size=5,
            marker=dict(
                size=10,
                color=(
                    (np.abs(infos.frequency) <= max_freq)
                    & (np.abs(infos.frequency) >= min_freq)
                ).astype("int"),
                # colorscale=[[0, "rgba(255,127,80, 0.01)"], [1, "rgba(255,127,80, 1)"]],
            ),
        )
    )
    fig.add_shape(
        type="circle",
        fillcolor="PaleTurquoise",
        line_color="LightSeaGreen",
        opacity=0.2,
        layer="below",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
    )
    fig.update_layout(
        margin={"l": 10, "b": 10, "t": 40, "r": 20},
        xaxis_title="Real Part",
        yaxis_title="Imaginary Part",
        width=400,
        height=400,
        # title='Eigenvalues',
        # template=PLOTLY_THEME,
    )
    return fig


def create_frequency_plot(infos, min_freq=None, max_freq=None):
    # infos is a dataframe with 'modulus' and 'angles' columns
    if min_freq is None:
        min_freq = infos["frequency"].min()
    if max_freq is None:
        max_freq = infos["frequency"].max()
    relevant_infos = infos[infos["frequency"] >= 0]
    fig = go.Figure(
        go.Scatter(
            x=relevant_infos["frequency"],
            y=relevant_infos["modulus"],  # plotting only positive frequencies
            mode="markers",
            marker_size=5,
            marker=dict(
                size=2,
                color=(
                    (np.abs(relevant_infos.frequency) <= max_freq)
                    & (np.abs(relevant_infos.frequency) >= min_freq)
                ).astype("int"),
                # colorscale=[[0, "rgba(255,127,80, 0.01)"], [1, "rgba(255,127,80, 1)"]],
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        # title='Frequency spectrum',
        margin={"l": 10, "b": 10, "t": 40, "r": 20},
        autosize=False,
        width=550,
        height=400,
        # template=PLOTLY_THEME,
    )
    max_range = np.max(infos["frequency"])
    fig.update_xaxes(
        range=[-0.1 * max_range, 1.1 * max_range]
    )  # Plotting with an outset for readability
    # version with amplitude multiplied with mode modulus

    return fig


# TODO: function comparing koopman to fourier


def create_plot_modes(infos, index=None, min_freq=None, max_freq=None):
    # image is false if data one dimensional, true if two dimensional
    # coordinates is hte position of each pixel if two dimensional
    # index is None if all modes are to be plotted, an array if some nodes must plotted or an int
    # if one mode must be plotted
    if index is None:
        index = infos["eig_num"].drop_duplicates().to_numpy()
    n_plots = len(index)

    if n_plots == 1:
        mode = infos[infos["eig_num"] == index[0]]
        freq = np.abs(mode["frequency"].to_numpy()[0])
        is_selected = int((freq <= max_freq) & (freq >= min_freq))
        opacity = is_selected + (1 - is_selected) * 0.2
        fig = px.scatter(x=mode["x"], y=mode["mode"].to_numpy().real, opacity=opacity)
        fig.update_layout(
            xaxis_title="Variables", yaxis_title="Value of mode", 
            # template=PLOTLY_THEME
        )
        return fig

    if n_plots % 2 == 0:
        fig = make_subplots(
            rows=n_plots // 2, cols=2, x_title="Variable", y_title="Signal"
        )
    else:
        fig = make_subplots(rows=n_plots // 2 + 1, cols=2)
    for i in index:
        mode = infos[infos["eig_num"] == i]
        freq = np.abs(mode["frequency"].to_numpy()[0])
        is_selected = int((freq <= max_freq) & (freq >= min_freq))
        opacity = is_selected + (1 - is_selected) * 0.2
        fig.add_trace(
            px.scatter(
                x=mode["x"], y=mode["mode"].to_numpy().real, opacity=opacity
            ).data[0],
            col=i % 2 + 1,
            row=i // 2 + 1,
        )
    fig.update_layout(
        title="Mode decomposition",
        # width=1100,
        height=700,
        # template=PLOTLY_THEME,
    )
    return fig


def create_2d_plot_modes(infos, index, min_freq, max_freq):
    if index is None:
        index = infos["eig_num"].drop_duplicates().to_numpy()
    n_plots = len(index)
    if n_plots % 2 == 0:
        fig = make_subplots(rows=n_plots // 2, cols=2)
    else:
        fig = make_subplots(rows=n_plots // 2 + 1, cols=2)
    for i in index:
        freq = infos[infos["eig_num"] == i]["frequency"].to_numpy()[0]
        is_selected = int((freq <= max_freq) & (freq >= min_freq))
        opacity = is_selected + (1 - is_selected) * 0.2
        fig.add_trace(
            px.scatter(
                infos[infos["eig_num"] == i],
                x="x",
                y="y",
                color=infos["mode"].to_numpy().real,
                opacity=opacity,
            ),
            col=i % 2,
            row=i // 2,
        )
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        # width=1100,
        height=700,
        # template=PLOTLY_THEME,
    )
    return fig


def create_combined_plot_modes(infos, T, min_freq, max_freq):
    preds = np.zeros(infos.var_index.max() + 1)
    for i in infos["eig_num"].unique():
        mode = infos[infos["eig_num"] == i]
        freq = mode["frequency"].to_numpy()[0]
        if (np.abs(freq) > max_freq) or (
            np.abs(freq) < min_freq
        ):  # if frequency outside selection, don't add the mode
            continue
        eigs = mode["eigval real"].unique()[0] + mode["eigval imag"].unique()[0] * 1j
        mode_value = mode["mode"].to_numpy()
        # summing the mode
        eigT = eigs**T
        preds += (eigT * mode_value).real
    fig = px.scatter(x=infos["x"].unique(), y=preds)
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Value",
        # width=500,
        height=500,
        # template=PLOTLY_THEME,
    )
    return fig


def create_combined_2d_plot_modes(infos, T, min_freq, max_freq):
    preds = np.zeros(infos.var_index.max() + 1)
    for i in infos["eig_num"].unique():
        mode = infos[infos["eig_num"] == i]
        freq = infos["frequency"].to_numpy()[0]
        if np.abs(freq) > max_freq or np.abs(freq) < min_freq:
            continue
        eigs = mode["eigval real"].unique()[0] + mode["eigval imag"].unique()[0] * 1j
        mode_value = mode["mode"].to_numpy()
        # summing the mode
        eigT = eigs**T
        preds += (eigT * mode_value).real
    fig = px.scatter(x=infos["x"].unique(), y=infos["y"].unique(), color=preds)
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Value",
        # width=700,
        height=500,
        # template=PLOTLY_THEME,
    )
    return fig


def map_predictions(preds, coordinates, height=False):
    # coordinates is a list of x,y coordinates in np array format
    if not height:
        fig = px.scatter(x=coordinates[:, 0], y=coordinates[:, 1], color=preds)
    else:
        fig = px.scatter_3d(
            x=coordinates[:, 0], y=coordinates[:, 1], z=preds, color=preds
        )
    return fig


def create_plot_pred(infos, preds):
    fig = px.line(y=preds, x=infos[infos["eig_num"] == 0]["x"])
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Predicted value",
        # template=PLOTLY_THEME,
    )
    return fig


# TODO: plot x= time, y= the mode frequencies, z = amplitudes


model_params = {
        'KernelDMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'Kernel', 'id': {'type': 'model-param', 'index': 'kernel'}, 'type': 'dropdown', 'options': ['DotProduct', 'RBF', 'Matern', '0.5*DotProduct + 0.5*RBF'], 'default': 'DotProduct'}, #'options': ['DotProduct', 'Exponentiation', 'PairwiseKernel', 'Sum', 'Product']
            {'label': 'Length Scale', 'id': {'type': 'model-param', 'index': 'length_scale'}, 'type': 'number', 'default': 1.0},
            # {'label': 'Optimal Sketching', 'id': {'type': 'model-param', 'index': 'optimal_sketching'}, 'type': 'checklist', 'default': False},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'DeepEDMD': [
            {'label': 'Maximum number of epochs', 'id': {'type': 'model-param', 'index': 'max_epochs'}, 'type': 'number', 'default': 10},
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'ExtendedDMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'FeatureMap', 'id': {'type': 'model-param', 'index': 'feature_map'}, 'type': 'dropdown', 'options': ['IdentityFeatureMap'], 'default': 'IdentityFeatureMap'},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
        'DMD': [
            {'label': 'Reduced Rank', 'id': {'type': 'model-param', 'index': 'reduced_rank'}, 'type': 'checklist', 'default': True},
            {'label': 'SVD Solver', 'id': {'type': 'model-param', 'index': 'svd_solver'}, 'type': 'dropdown', 'options': ['full', 'arnoldi', 'randomized'], 'default': 'full'},
            {'label': 'Iterated Power', 'id': {'type': 'model-param', 'index': 'iterated_power'}, 'type': 'number', 'default': 1},
            # {'label': 'N Oversamples', 'id': {'type': 'model-param', 'index': 'n_oversamples'}, 'type': 'number', 'default': 5},
            {'label': 'RNG Seed', 'id': {'type': 'model-param', 'index': 'rng_seed'}, 'type': 'number', 'default': None}
        ],
    }

def create_model_params(model_name):
    params = model_params.get(model_name, [])

    children = []
    for param in params:
        if param['type'] == 'number':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id=param['id'], type="number", value=param['default'], placeholder=param['label']),
            ], className="p-2"))
        elif param['type'] == 'dropdown':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Dropdown(id=param['id'], options=[{'label': val, 'value': val} for val in param['options']], value=param['default'], clearable=False),
            ], className="p-2"))
        elif param['type'] == 'checklist':
            children.append(html.Div([
                dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.RadioItems(id=param['id'], options=['True', 'False'], value=str(param['default']), inline=True),
            ], className="p-2"))

    return children



# Dictionary of dataset parameters
dataset_params = {
    'LinearModel': [
        {'label': 'Noise', 'id': {'type': 'dynamic-param', 'index': 'noise'}, 'value': 1., 'step': 0.01},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1},
        {'label': 'Number features', 'id': {'type': 'dynamic-param', 'index': 'num_features'}, 'value': 10, 'step': 1},
        {'label': 'Rang de la matrice', 'id': {'type': 'dynamic-param', 'index': 'r'}, 'value': 5, 'step': 1},
        {'label': 'Lambda', 'id': {'type': 'dynamic-param', 'index': 'l'}, 'value': 1., 'step': 0.1}
    
    ],
    'LogisticMap': [
        {'label': 'r', 'id': {'type': 'dynamic-param', 'index': 'r'}, 'value': 4.0, 'step': 0.1},
        {'label': 'N', 'id': {'type': 'dynamic-param', 'index': 'N'}, 'value': None, 'step': 1},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1}
    ],
    'LangevinTripleWell1D': [
        {'label': 'gamma', 'id': {'type': 'dynamic-param', 'index': 'gamma'}, 'value': 0.1, 'step': 0.01},
        {'label': 'kt', 'id': {'type': 'dynamic-param', 'index': 'kt'}, 'value': 1.0, 'step': 0.1},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 1e-4, 'step': 1e-5},
        {'label': 'RNG Seed', 'id': {'type': 'dynamic-param', 'index': 'rng_seed'}, 'value': None, 'step': 1}
    ],
    'DuffingOscillator': [
        {'label': 'alpha', 'id': {'type': 'dynamic-param', 'index': 'alpha'}, 'value': 0.5, 'step': 0.01},
        {'label': 'beta', 'id': {'type': 'dynamic-param', 'index': 'beta'}, 'value': 0.0625, 'step': 0.0001},
        {'label': 'gamma', 'id': {'type': 'dynamic-param', 'index': 'gamma'}, 'value': 0.1, 'step': 0.1},
        {'label': 'delta', 'id': {'type': 'dynamic-param', 'index': 'delta'}, 'value': 2.5, 'step': 0.1},
        {'label': 'omega', 'id': {'type': 'dynamic-param', 'index': 'omega'}, 'value': 2.0, 'step': 0.1},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 0.01, 'step': 0.01}
    ],
    'Lorenz63': [
        {'label': 'sigma', 'id': {'type': 'dynamic-param', 'index': 'sigma'}, 'value': 10, 'step': 1},
        {'label': 'mu', 'id': {'type': 'dynamic-param', 'index': 'mu'}, 'value': 28, 'step': 1},
        {'label': 'beta', 'id': {'type': 'dynamic-param', 'index': 'beta'}, 'value': 8 / 3, 'step': 0.01},
        {'label': 'dt', 'id': {'type': 'dynamic-param', 'index': 'dt'}, 'value': 0.01, 'step': 0.01}
    ]
}

def create_dataset_params(dataset_name):
    # Get parameters for the selected dataset
    params = dataset_params.get(dataset_name, [])
    
    # Create input components for each parameter
    children = []
    for param in params:
        children.append(html.Div([
            dbc.Label(f"{param['label']}: ", style={"display": "inline-block", "margin-right": "8px"},),
            dcc.Input(id=param['id'], type="number", value=param['value'], step=param['step'], placeholder=param['label']),
        ], className="p-2"))

    return children


def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in content_type:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            raise ValueError('Unsupported file format')
        print(df)
        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])