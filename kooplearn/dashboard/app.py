import numpy as np
import pandas as pd
from dash import Dash, Input, Output, callback, dcc, html, no_update, callback_context
from dash.dependencies import ALL, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px

from sklearn.gaussian_process.kernels import *
from kooplearn.models.feature_maps import *
from featuremap_example import feature_map

from kooplearn.dashboard.visualizer import Visualizer
from kooplearn.datasets import *
from kooplearn.models import *
from kooplearn.data import traj_to_contexts
from kooplearn.dashboard.utils import create_model_params, create_dataset_params, parse_contents

from scipy.stats import ortho_group


# stylesheet with the .dbc class to style  dcc, DataTable and AG Grid components with a Bootstrap theme
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc_css])

header = html.H4(
    "Dash web application for visualisation of a Koopman operator", className="bg-primary text-white p-2 mb-2 text-center"
)

progress_bar = dbc.Progress(id="progress-bar", striped=True, animated=True, style={"height": "20px", "visibility": "hidden"}, value=100)

update_button = html.Div([dbc.Button("Update Plots", id="update-button", className="mb-3", color="primary", n_clicks=1), progress_bar])


# Define your controls using Dash Bootstrap Components
models = dbc.Tab([html.Div(
            [
                dbc.Label("Select a model"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "KernelDMD", "value": "KernelDMD"},
                        {"label": "DeepEDMD", "value": "DeepEDMD"},
                        {"label": "DMD", "value": "DMD"},
                        {"label": "ExtendedDMD", "value": "ExtendedDMD"},                        
                    ],
                    value="KernelDMD",  # Default model selection
                    clearable=False
                ),],
                className="p-3",),
                html.Div([
                dbc.Label("Rank: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="rank-input", type="number", min=1, step=1, placeholder="rank", value=1),], className="p-3"),  # Assuming rank starts at 1 and increments by 1
                
                html.Div([
                dbc.Label("Tikhonov Regularization: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="tikhonov_reg-input", type="number", placeholder="tikhonov_reg"),], className="p-3",),

                html.Div([
                dbc.Label("context_window_len: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="context_window_len-input", type="number", min=2, step=1, value=2, placeholder="context_window_len"),], className="p-3",),

                html.Div(id='model-params-div'),
        ],
        label="Models"
        )


file_upload = dcc.Upload(
    id='upload-data',
    children=html.Div(id='upload-text', children=[
        'Drag and Drop or ',
        html.A('Select a CSV File')
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=False
)


datasets = dbc.Tab([html.Div(
            [
                file_upload,

                dbc.Label("Select a predefined dataset"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[
                        {"label": "LinearModel", "value": "LinearModel"},
                        {"label": "LogisticMap", "value": "LogisticMap"},
                        {"label": "LangevinTripleWell1D", "value": "LangevinTripleWell1D"},
                        {"label": "DuffingOscillator", "value": "DuffingOscillator"},
                        {"label": "Lorenz63", "value": "Lorenz63"},
                    ],
                    value="LinearModel",
                    clearable=False
                ),
                html.Div(id='dataset-params-div'),  # This Div will be populated with inputs dynamically
        
                dbc.Label("T parameter of the sample function: ", style={"display": "inline-block", "margin-right": "8px"}),
                dcc.Input(id="T_sample-input", type="number", min=1, step=1, value=100, placeholder="T"),
            ],
            className="p-3",
        )],
        label="Datasets"
        )

slider1 = html.Div(
            [
                dbc.Label("Frequency"),
                dcc.RangeSlider(
                    min=-0.1,
                    max= 1, #int(viz.infos["frequency"].max()) + 1,
                    marks= {"0": "0", "1": "1"}, #frequency_dict,
                    id="freq_range_slider",
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="p-0",
                ),
            ],
            className="p-3",
)

slider2 = html.Div(
            [
                dcc.Input(id="Tmax", type="number", placeholder="input T max"),
                dcc.Slider(min=0, 
                           max=1, 
                           step=1, 
                           id="T",
                           tooltip={"placement": "bottom", "always_visible": True},
                           className="p-0",
                    ),
            ],
            className="p-3",
)

modes = dbc.Tab([html.Div(
            [
                html.H4("Modes"),
                dbc.Label("Select a mode"),
                dcc.Dropdown(
                    id="modes_select",
                    options=[{"label": str(i), "value": str(i)} for i in range(10)],  # Placeholder values
                    value="All",  # Placeholder value
                    clearable=False
                ),
                dcc.Graph(id="modes-plot"),
                # html.H1("Prediction"),
                # dcc.Graph(id='pred-plot', figure=viz.plot_preds())
            ],
            className="p-3",
        )], label="Modes")

controls = dbc.Tab([html.Div(
    [slider1, slider2],
    className="p-3",
)],label="Frequency and T")


tabs_control = dbc.Card(dbc.Tabs([controls, models, datasets]), body=True,)

graph1 = dbc.Col([html.Div(
            [html.H4("Eigenvalues plot"),
                dcc.Graph(id="eig-plot"),
                ],
                )], align="center", width=5)                


graph2 = dbc.Col([html.Div(
            [
                html.H4("Frequency plot"),
                dcc.Graph(id="freq-plot"),
                ],
                )], align="center", width=7)



dimension_selector = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label('Select Dimension 1 (x-axis):'),
            dcc.Dropdown(id='dimension-dropdown-x', options=[], value=None),
        ], width=6),
        dbc.Col([
            html.Label('Select Dimension 2 (x-axis):'),
            dcc.Dropdown(id='dimension-dropdown-y', options=[], value=None),
        ], width=6),
    ]),
], className="p-3")


dataset_plot = dbc.Tab([html.Div(
            [
                dimension_selector,
                dcc.Graph(id="dataset-plot"),
            ],
            className="p-3",
        )], label="Dataset Plot")

hidden_div = html.Div(id='intermediate-value', style={'display': 'none'})

tab_plots = dbc.Tab(dbc.Row([graph1, graph2]), label="Plots")

tabs_graphs = dbc.Card(dbc.Tabs([dataset_plot, tab_plots, modes]), body=True)


app.layout = dbc.Container(
    [
        header,
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([html.Img(src="https://kooplearn.readthedocs.io/latest/_static/logo.svg", height="120px")], align="center"),
                    dbc.Col([update_button], align="center"),
                    ]),
                tabs_control
            ],  width=4),
            dbc.Col([tabs_graphs], width=8),
        ]),
        html.Div(id="click-count", style={"display": "none"}, children="1"),

        hidden_div,
    ], 
    fluid=True,
    className="bg-light",
)



@app.callback(
    [
        Output('intermediate-value', 'children'),
        Output('dataset-dropdown', 'value'),
    ],
    [
        Input('upload-data', 'contents'),
    ],
    [
        State('upload-data', 'filename'),
    ],
    prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate

    df = parse_contents(contents)
    if df is None:
        return None, no_update

    # Assuming all columns are relevant
    return df.to_json(date_format='iso', orient='split'), None


@app.callback(
    [
        Output('upload-text', 'children'),
        Output('upload-data', 'style')
    ],
    [Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def update_upload_text_and_style(contents):
    if contents is not None:
        # Changes when file is uploaded
        return "File Uploaded", {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'backgroundColor': '#90ee90'  # Light green background
        }
    else:
        # Default state
        return ['Drag and Drop or ', html.A('Select a CSV File')], {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }




@app.callback(
    Output('dataset-params-div', 'children'),
    Input('dataset-dropdown', 'value')
)
def update_dataset_params(selected_dataset):
    return create_dataset_params(selected_dataset)


@app.callback(
    Output('model-params-div', 'children'),
    Input('model-dropdown', 'value')
)
def update_model_params(selected_model):
    return create_model_params(selected_model)

@app.callback(
    Output("click-count", "children"),
    Output("progress-bar", "style", allow_duplicate=True),
    Input("update-button", "n_clicks"),
    State("click-count", "children"),
    prevent_initial_call=True
)
def update_click_count_and_show_progress(n_clicks, click_count):
    if n_clicks is None:
        raise PreventUpdate

    # Increment click count
    new_click_count = str(int(click_count) + 1)

    # Show progress bar
    progress_style = {"height": "20px", "visibility": "visible"}

    return new_click_count, progress_style

@app.callback(
    [Output('dimension-dropdown-x', 'value'),
     Output('dimension-dropdown-y', 'value')],
    [Input('dataset-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_dimension_dropdowns(dataset):
    return None, None



@callback(
    [
        Output("eig-plot", "figure"),
        Output("freq-plot", "figure"),
        Output("modes-plot", "figure"),
        Output('upload-data', 'contents'),
        Output("dataset-plot", "figure"),
        Output("dimension-dropdown-x", "options"),  # Dynamically set options
        Output("dimension-dropdown-y", "options"),  # Dynamically set options
        Output("T", "max"),
        Output("freq_range_slider", "max"),
        Output("freq_range_slider", "marks"),
        Output("modes_select", "options"),
        Output("rank-input", "value"),
        Output("tikhonov_reg-input", "value"),
        Output("context_window_len-input", "value"),
        Output("progress-bar", "style"),
    ],  

    [Input("click-count", "children"),],

    [
        State("dimension-dropdown-x", "value"),
        State("dimension-dropdown-y", "value"),
        State("freq_range_slider", "value"),
        State("Tmax", "value"),
        State("T", "value"),
        State("modes_select", "value"),
        State("rank-input", "value"),
        State("tikhonov_reg-input", "value"),
        State({'type': 'model-param', 'index': ALL}, 'value'),
        State("context_window_len-input", "value"),
        State({'type': 'dynamic-param', 'index': ALL}, 'value'),
        State("T_sample-input", "value"),
        State('intermediate-value', 'children'),
        State("dataset-dropdown", "value"),
        State("model-dropdown", "value"),
    ],

)
def main(click_count, dim_x, dim_y, value, Tmax, T, mode_selection, rank, tikhonov_reg, model_dynamic_params,
                       context_window_len, dynamic_params, T_sample, json_data,
                       selected_dataset="LinearModel", selected_model="KernelDMD"):

    if click_count == "0":
        # Prevent update before the app loads (no button click yet)
        raise PreventUpdate

    if json_data is not None and selected_dataset is None:
        df = pd.read_json(json_data, orient='split')
        _Z = df.values
        X = traj_to_contexts(_Z,  context_window_len=context_window_len)
    
    else:
        # print(dynamic_params)
        if selected_dataset == "LinearModel":
            # np.random.seed(10)
            # H = ortho_group.rvs(10)
            # eigs = np.exp(-np.arange(10))
            # A = H @ (eigs * np.eye(10)) @ H.T
            noise=1.
            rng_seed=None

            num_features=10
            r=5
            l=1

            if dynamic_params!=[]:
                noise = dynamic_params[0]
                rng_seed = dynamic_params[1]

                num_features = dynamic_params[2]
                r = dynamic_params[3]
                l = dynamic_params[4]
            
            if rng_seed is not None:
                np.random.seed(rng_seed)

            f = lambda x, r : 1. if x<r else 0.
            H = ortho_group.rvs(num_features) 
            eigs = np.array([l*f(i, r) for i in range(num_features)]) 
            A = H @ (eigs * np.eye(num_features)) @ H.T 

            dataset = LinearModel(A = A, noise=noise, rng_seed=rng_seed)
            _Z = dataset.sample(X0 = np.zeros(A.shape[0]), T = T_sample)
            X = traj_to_contexts(_Z, context_window_len=context_window_len)

        elif selected_dataset == "LogisticMap":
            r_param=4.0
            N_param=None
            rng_seed=None
            if dynamic_params!=[]:
                r_param = dynamic_params[0]
                N_param = dynamic_params[1]
                rng_seed = dynamic_params[2]
            dataset = LogisticMap(r=r_param, N=N_param, rng_seed=rng_seed)
            _Z = dataset.sample(0.2, T_sample)
            X = traj_to_contexts(_Z,  context_window_len=context_window_len)

        elif selected_dataset == "LangevinTripleWell1D":
            gamma=0.1
            kt=1.0
            dt=1e-4
            rng_seed=None
            if dynamic_params!=[]:
                gamma = dynamic_params[0]
                kt = dynamic_params[1]
                dt = dynamic_params[2]
                rng_seed = dynamic_params[3]
            dataset = LangevinTripleWell1D(gamma=gamma, kt=kt, dt=dt, rng_seed=rng_seed)
            _Z = dataset.sample(0., T_sample)
            X = traj_to_contexts(_Z,  context_window_len=context_window_len)

        elif selected_dataset == "DuffingOscillator":
            alpha=0.5
            beta=0.0625
            gamma=0.1
            delta=2.5
            omega=2.0
            dt=0.01
            if dynamic_params!=[]:
                alpha = dynamic_params[0]
                beta = dynamic_params[1]
                gamma = dynamic_params[2]
                delta = dynamic_params[3]
                omega = dynamic_params[4]
                dt = dynamic_params[5]
            dataset = DuffingOscillator(alpha=alpha, beta=beta, gamma=gamma, delta=delta, omega=omega, dt=dt)
            _Z = dataset.sample(np.array([0.,0.]), T_sample)
            X = traj_to_contexts(_Z,  context_window_len=context_window_len)

            # import pandas as pd
            # df = pd.DataFrame({'dim0': _Z[:, 0],
            #            'dim1': _Z[:, 1]})
            # df.to_csv('dataset_example.csv', index=False) 

        elif selected_dataset == "Lorenz63":
            sigma=10
            mu=28
            beta=8 / 3
            dt=0.01
            if dynamic_params!=[]:
                sigma = dynamic_params[0]
                mu = dynamic_params[1]
                beta = dynamic_params[2]
                dt = dynamic_params[3]
            dataset = Lorenz63(sigma=sigma, mu=mu, beta=beta, dt=dt)
            _Z = dataset.sample(np.array([0,0.1,0]), T_sample)    
            X = traj_to_contexts(_Z,  context_window_len=context_window_len)

    # print(_Z)
    # print(len(_Z))
    # print(len(_Z[0]))

    num_dimensions = len(_Z[0])
    dimension_options = [{'label': f'variable {i}', 'value': i} for i in range(num_dimensions)]

    if dim_x is not None and dim_y is not None:
        # print(_Z[:, dim_x])
        # print(_Z[:, dim_y])

        fig_dataset = px.line(
            x=_Z[:, dim_x], 
            y=_Z[:, dim_y], 
            labels={"x": f"variable {dim_x}", "y": f"variable {dim_y}"}
        )
        fig_dataset.update_layout(title="Dataset Plot", xaxis_title=f"variable {dim_x}", yaxis_title=f"variable {dim_y}")

    else:
        fig_dataset = px.line(_Z, labels={"index":"time"})

    if selected_model == "KernelDMD":
        operator_kwargs = {'kernel': DotProduct()}
        if dynamic_params!=[]:
            kernel_mapping = {
            'DotProduct': DotProduct(),
            'RBF': RBF(length_scale=model_dynamic_params[4]), 
            'Matern': Matern(length_scale=model_dynamic_params[4]), 
            '0.5*DotProduct + 0.5*RBF': 0.5*DotProduct() + 0.5*RBF(length_scale=model_dynamic_params[4])
        }
            operator_kwargs = {
                'kernel': kernel_mapping.get(model_dynamic_params[3], DotProduct()),
                'reduced_rank': model_dynamic_params[0],
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                # 'optimal_sketching': model_dynamic_params[5],
                'rng_seed': model_dynamic_params[5] if model_dynamic_params[5] != '' else None  # Handle empty string for None
            }
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg                        
        operator = KernelDMD(**operator_kwargs)

    elif selected_model == "DeepEDMD":
        operator_kwargs = {'feature_map': feature_map(data=X, context_window_len=context_window_len)}
        if dynamic_params!=[]:
            operator_kwargs = {
                'feature_map': feature_map(data=X, context_window_len=context_window_len, max_epochs=model_dynamic_params[0]),
                'reduced_rank': model_dynamic_params[1],
                'svd_solver': model_dynamic_params[2],
                'iterated_power': model_dynamic_params[3],
                # 'n_oversamples': model_dynamic_params[4],
                'rng_seed': model_dynamic_params[4] if model_dynamic_params[4] != '' else None  # Handle empty string for None
            }
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg    
        operator = DeepEDMD(**operator_kwargs)

    elif selected_model == "ExtendedDMD":
        FeatureMap_mapping = {
            'IdentityFeatureMap': IdentityFeatureMap(),
        }
        operator_kwargs = {}
        if dynamic_params!=[]:
            operator_kwargs = {
                'feature_map': FeatureMap_mapping.get(model_dynamic_params[3], IdentityFeatureMap()),
                'reduced_rank': model_dynamic_params[0],
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                'rng_seed': model_dynamic_params[4] if model_dynamic_params[4] != '' else None  # Handle empty string for None
            }
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg    
        operator = ExtendedDMD(**operator_kwargs)

    elif selected_model == "DMD":
        operator_kwargs = {}
        if dynamic_params!=[]:
            operator_kwargs = {
                'reduced_rank': model_dynamic_params[0],
                'svd_solver': model_dynamic_params[1],
                'iterated_power': model_dynamic_params[2],
                # 'n_oversamples': model_dynamic_params[3],
                'rng_seed': model_dynamic_params[3] if model_dynamic_params[3] != '' else None  # Handle empty string for None
            }
        operator_kwargs["rank"] = rank
        if tikhonov_reg is not None:
            operator_kwargs["tikhonov_reg"] = tikhonov_reg
        operator = DMD(**operator_kwargs)

    operator.fit(X)
    viz = Visualizer(koopman=operator)
    available_modes = viz.infos["eig_num"].unique().astype("str")
    available_modes = np.insert(available_modes, 0, "All")
    available_modes = np.insert(available_modes, 1, "Combined")

    frequencies = viz.infos["frequency"].unique()
    pos_frequencies = frequencies[frequencies > 0]
    frequency_dict = {i: str(round(i, 3)) for i in pos_frequencies}
    frequency_dict[0] = "0"

    # Update the modes_select dropdown options
    modes_select_options = [{"label": str(i), "value": str(i)} for i in available_modes]

    if value is None:
        min_freq = viz.infos["frequency"].unique().min()
        max_freq = viz.infos["frequency"].unique().max()
    else:
        min_freq = value[0]
        max_freq = value[1]

    fig_eig = viz.plot_eigs(min_freq, max_freq)
    fig_freqs = viz.plot_freqs(min_freq, max_freq)       

    if T is None:
        T = 1

    if mode_selection == "All":
        fig_modes = viz.plot_modes(index=None, min_freq=min_freq, max_freq=max_freq)
    elif mode_selection == "Combined":
        fig_modes = viz.plot_combined_modes(T, min_freq, max_freq)
    else:
        fig_modes = viz.plot_modes(
            index=[int(mode_selection)], min_freq=min_freq, max_freq=max_freq
        )
    # fig_pred = viz.plot_preds(operator.X_fit_[-1], 1, min_freq, max_freq)

    progress_style = {"height": "20px", "visibility": "hidden"}

    return (fig_eig, fig_freqs, fig_modes, None, fig_dataset, dimension_options, dimension_options, Tmax, 
            int(viz.infos["frequency"].max()) + 1, frequency_dict, 
            modes_select_options, rank, tikhonov_reg, context_window_len, progress_style)


app.run_server(debug=True)
