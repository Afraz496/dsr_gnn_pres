import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
import json
import os

# Load data if the file exists, otherwise use mock data
def load_results():
    try:
        data = np.load('model_results.npz')
        return {
            'true_values': data['true_values'],
            'catboost_predictions': data['catboost_predictions'],
            'mpnn_predictions': data['mpnn_predictions'],
            'node_indices': data['node_indices'],
            'timestamps': data['timestamps']
        }
    except FileNotFoundError:
        print("Results file not found, using mock data")
        # Generate mock data
        n_samples = 1000
        n_nodes = 129
        max_timestamp = n_samples // n_nodes
        
        node_indices = np.array([i % n_nodes for i in range(n_samples)])
        timestamps = np.array([i // n_nodes for i in range(n_samples)])
        true_values = np.random.normal(50, 20, n_samples)
        catboost_predictions = true_values + np.random.normal(0, 5, n_samples)
        mpnn_predictions = true_values + np.random.normal(0, 3, n_samples)
        
        return {
            'true_values': true_values,
            'catboost_predictions': catboost_predictions,
            'mpnn_predictions': mpnn_predictions,
            'node_indices': node_indices,
            'timestamps': timestamps
        }

# Load England geography data or generate mock graph
def load_graph_data():
    try:
        # Try to load real geographic data
        with open('england_regions.json', 'r') as f:
            region_data = json.load(f)
        G = nx.Graph()
        # Add nodes and edges based on real geography
        # ...
    except FileNotFoundError:
        # Generate mock graph
        print("Geography data not found, generating mock graph")
        G = nx.random_geometric_graph(129, 0.125)
        # Add random positions if not created by the geometric graph
        for node in G.nodes():
            if 'pos' not in G.nodes[node]:
                G.nodes[node]['pos'] = (np.random.random(), np.random.random())
    return G

# Get metrics
def calculate_metrics(true_values, predictions):
    mse = np.mean((true_values - predictions) ** 2)
    mae = np.mean(np.abs(true_values - predictions))
    # R² calculation
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    ss_res = np.sum((true_values - predictions) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  # Avoid division by zero
    return {
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'R²': round(r2, 4)
    }

# Load the data
results_data = load_results()
graph = load_graph_data()

# Prepare node data for visualization
def prepare_node_data(timestamp, values=None):
    if values is None:
        # Use true values by default
        mask = results_data['timestamps'] == timestamp
        node_values = {}
        for idx, node_idx in enumerate(results_data['node_indices'][mask]):
            node_values[int(node_idx)] = results_data['true_values'][mask][idx]
    else:
        # Use provided values
        mask = results_data['timestamps'] == timestamp
        node_values = {}
        for idx, node_idx in enumerate(results_data['node_indices'][mask]):
            node_values[int(node_idx)] = values[mask][idx]
    
    return node_values

# Get unique timestamps
unique_timestamps = np.unique(results_data['timestamps'])
max_timestamp = int(np.max(unique_timestamps))

# Calculate metrics
catboost_metrics = calculate_metrics(
    results_data['true_values'],
    results_data['catboost_predictions']
)
mpnn_metrics = calculate_metrics(
    results_data['true_values'],
    results_data['mpnn_predictions']
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("England COVID-19 Prediction Dashboard", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Metrics"),
                dbc.CardBody([
                    html.Div([
                        html.H5("CatBoost Model"),
                        dbc.Row([
                            dbc.Col([
                                html.P(f"MSE: {catboost_metrics['MSE']}")
                            ]),
                            dbc.Col([
                                html.P(f"MAE: {catboost_metrics['MAE']}")
                            ]),
                            dbc.Col([
                                html.P(f"R²: {catboost_metrics['R²']}")
                            ])
                        ])
                    ]),
                    html.Hr(),
                    html.Div([
                        html.H5("MPNN-LSTM Model"),
                        dbc.Row([
                            dbc.Col([
                                html.P(f"MSE: {mpnn_metrics['MSE']}")
                            ]),
                            dbc.Col([
                                html.P(f"MAE: {mpnn_metrics['MAE']}")
                            ]),
                            dbc.Col([
                                html.P(f"R²: {mpnn_metrics['R²']}")
                            ])
                        ])
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Temporal Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Timestamp:"),
                            dcc.Slider(
                                id='timestamp-slider',
                                min=0,
                                max=max_timestamp,
                                value=0,
                                marks={i: str(i) for i in range(0, max_timestamp + 1, 5)},
                                step=1
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Button("⏪", id="btn-step-back", color="secondary", className="me-2"),
                                dbc.Button("⏯️", id="btn-play-pause", color="primary", className="me-2"),
                                dbc.Button("⏩", id="btn-step-forward", color="secondary", className="me-2"),
                                dbc.Button("Reset", id="btn-reset", color="secondary"),
                            ], className="d-flex justify-content-center mt-3")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Animation Speed:"),
                            dcc.Slider(
                                id='speed-slider',
                                min=500,
                                max=3000,
                                value=1000,
                                marks={500: 'Fast', 1500: 'Medium', 3000: 'Slow'},
                                step=100
                            )
                        ])
                    ]),
                    dcc.Interval(
                        id='interval-component',
                        interval=1000,  # in milliseconds
                        n_intervals=0,
                        disabled=True
                    )
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Graph Visualization"),
                dbc.CardBody([
                    html.Div([
                        dcc.Graph(id='graph-viz', style={'height': '500px'})
                    ])
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Predictions"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Select Node:"),
                        dcc.Dropdown(
                            id='node-selector',
                            options=[{'label': f'Node {i}', 'value': i} for i in range(129)],
                            value=0
                        )
                    ]),
                    html.Div([
                        dcc.Graph(id='prediction-chart')
                    ])
                ])
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time Series Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='time-series-chart')
                ])
            ], className="mb-4")
        ])
    ])
], fluid=True)

@app.callback(
    Output('interval-component', 'disabled'),
    Output('btn-play-pause', 'children'),
    Input('btn-play-pause', 'n_clicks'),
    State('interval-component', 'disabled')
)
def toggle_play(n_clicks, is_disabled):
    if n_clicks is None:
        return True, "⏯️"
    return not is_disabled, "⏸️" if is_disabled else "▶️"

@app.callback(
    Output('interval-component', 'interval'),
    Input('speed-slider', 'value')
)
def update_speed(value):
    return value

@app.callback(
    Output('timestamp-slider', 'value'),
    Input('interval-component', 'n_intervals'),
    Input('btn-step-forward', 'n_clicks'),
    Input('btn-step-back', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    State('timestamp-slider', 'value')
)
def update_timestamp(n_intervals, forward_clicks, back_clicks, reset_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'interval-component':
        new_value = current_value + 1
        if new_value > max_timestamp:
            return 0  # Loop back to start
        return new_value
    elif trigger_id == 'btn-step-forward':
        new_value = current_value + 1
        if new_value > max_timestamp:
            return max_timestamp
        return new_value
    elif trigger_id == 'btn-step-back':
        new_value = current_value - 1
        if new_value < 0:
            return 0
        return new_value
    elif trigger_id == 'btn-reset':
        return 0
    
    return current_value

@app.callback(
    Output('graph-viz', 'figure'),
    Input('timestamp-slider', 'value')
)
def update_graph(timestamp):
    # Get node values for this timestamp
    node_values = prepare_node_data(timestamp)
    
    # Prepare graph layout
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        pos = nx.spring_layout(graph, seed=42)
    
    # Normalize values for color mapping
    values = list(node_values.values())
    if not values:
        # If no values for this timestamp, use defaults
        vmin, vmax = 0, 100
    else:
        vmin, vmax = min(values), max(values)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        value = node_values.get(node, 0)
        node_color.append(value)
        node_text.append(f'Node {node}: {value:.2f}')
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Value',
                xanchor='left'
                # Removed titleside parameter that was causing the error
            ),
            line_width=2,
            cmin=vmin,
            cmax=vmax
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f'England COVID-19 Graph at Timestamp {timestamp}', 
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='rgba(0,0,0,0)'
                ))
    
    return fig


@app.callback(
    Output('prediction-chart', 'figure'),
    Input('node-selector', 'value'),
    Input('timestamp-slider', 'value')
)
def update_prediction_chart(node_id, timestamp):
    # Filter data for the selected node
    node_mask = results_data['node_indices'] == node_id
    time_data = results_data['timestamps'][node_mask]
    true_data = results_data['true_values'][node_mask]
    catboost_data = results_data['catboost_predictions'][node_mask]
    mpnn_data = results_data['mpnn_predictions'][node_mask]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_data, y=true_data,
        mode='lines+markers',
        name='True Values',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=time_data, y=catboost_data,
        mode='lines+markers',
        name='CatBoost Predictions',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=time_data, y=mpnn_data,
        mode='lines+markers',
        name='MPNN-LSTM Predictions',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f"Node {node_id} Predictions Over Time (Timestamp {timestamp})",
        xaxis_title="Timestamp",
        yaxis_title="Prediction Value",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(b=20, l=40, r=40, t=40)
    )
    
    return fig

@app.callback(
    Output('time-series-chart', 'figure'),
    Input('timestamp-slider', 'value')
)
def update_time_series_chart(timestamp):
    # Aggregate true values for the selected timestamp
    timestamp_mask = results_data['timestamps'] == timestamp
    time_series_data = results_data['true_values'][timestamp_mask]
    
    # Create the figure for the time series chart
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=time_series_data,
        name='True Values Distribution',
        marker=dict(color='blue', opacity=0.6),
        nbinsx=50
    ))
    
    fig.update_layout(
        title=f"True Values Distribution at Timestamp {timestamp}",
        xaxis_title="True Value",
        yaxis_title="Count",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(b=20, l=40, r=40, t=40)
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)