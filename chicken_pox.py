import torch
import torch_geometric_temporal
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_leaflet as dl
import datetime

# Define a simple GNN model for predictions
class GCNModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def normalize_coordinates(lat, lon, width=500, height=500):
    # Normalize latitude and longitude to fit in a given width and height
    min_lat, max_lat = 45.74, 48.58
    min_lon, max_lon = 16.11, 22.90

    x = (lon - min_lon) / (max_lon - min_lon) * width
    y = (max_lat - lat) / (max_lat - min_lat) * height
    return {'x': x, 'y': y}

# Load raw data
def load_raw_data():
    df = pd.read_csv('hungary_chickenpox.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# County mapping as per CSV columns
csv_counties = ['BUDAPEST','BARANYA','BACS','BEKES','BORSOD','CSONGRAD','FEJER','GYOR','HAJDU',
                'HEVES','JASZ','KOMAROM','NOGRAD','PEST','SOMOGY','SZABOLCS','TOLNA','VAS','VESZPREM','ZALA']

# Mapping for visualization
county_coordinates = {
    'BACS': (46.5935, 19.3547), 'BARANYA': (45.9910, 18.2232), 'BEKES': (46.7639, 21.0845),
    'BORSOD': (48.2286, 20.6180), 'CSONGRAD': (46.4167, 20.2500), 'FEJER': (47.1130, 18.4367),
    'GYOR': (47.6849, 17.2610), 'HAJDU': (47.5160, 21.6500), 'HEVES': (47.8500, 20.0833),
    'JASZ': (47.1667, 20.4167), 'KOMAROM': (47.6833, 18.3333), 'NOGRAD': (48.0000, 19.5000),
    'PEST': (47.5000, 19.3333), 'SOMOGY': (46.5833, 17.6667), 'SZABOLCS': (47.9000, 22.0000),
    'TOLNA': (46.5000, 18.5000), 'VAS': (47.0833, 16.5667), 'VESZPREM': (47.1000, 17.9000),
    'ZALA': (46.8333, 16.8333), 'BUDAPEST': (47.497913, 19.040236)
}

def get_dataset_snapshots(dataset):
    """Convert the iterable dataset to a list of snapshots"""
    snapshots = []
    for snapshot in dataset:
        snapshots.append(snapshot)
    return snapshots

def train_gnn_model(dataset):
    """Train a simple GNN model on the dataset"""
    snapshots = get_dataset_snapshots(dataset)
    train_split = int(len(snapshots) * 0.7)
    
    # Initialize model
    node_features = snapshots[0].x.shape[1]
    model = GCNModel(node_features=node_features, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    for epoch in range(50):  # Train for 50 epochs
        for i in range(train_split - 1):  # Use t to predict t+1
            snapshot = snapshots[i]
            y_true = snapshots[i+1].x[:, 0].reshape(-1, 1)  # Target is next snapshot's feature
            
            optimizer.zero_grad()
            out = model(snapshot.x, snapshot.edge_index)
            loss = F.mse_loss(out, y_true)
            loss.backward()
            optimizer.step()
    
    # Generate predictions for all snapshots
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for snapshot in snapshots:
            pred = model(snapshot.x, snapshot.edge_index)
            # Ensure predictions are non-negative
            pred = torch.clamp(pred, min=0)
            predictions.append(pred.numpy().flatten())
    
    return predictions

def create_dashboard(dataset, raw_data):
    # Convert dataset to a list of snapshots for direct indexing
    snapshots = get_dataset_snapshots(dataset)
    total_steps = len(snapshots)
    
    # Train model and get predictions
    predictions = train_gnn_model(dataset)
    print(predictions)
    app = dash.Dash(__name__)
    time_labels = generate_time_labels(start_year=2005, num_steps=total_steps)

    positions = {county: normalize_coordinates(lat, lon) for county, (lat, lon) in county_coordinates.items()}

    step_size = max(1, total_steps // 10)
    time_slider_marks = {i: time_labels[i] for i in range(0, total_steps, step_size) if i < total_steps}

    app.layout = html.Div([ 
        html.H1("Hungary Chickenpox Cases - Temporal GNN Visualization", style={'textAlign': 'center'}),

        html.Div([
            html.Div(
                dl.Map(
                    [
                        dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                        dl.LayerGroup(id='markers')
                    ],
                    center=[47.1625, 19.5033],
                    zoom=7,
                    style={'width': '100%', 'height': '500px'}, 
                    id='map'
                ),
                style={'width': '50%', 'display': 'inline-block'}
            ),

            html.Div(
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=[],
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '500px'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'background-color': 'data(color)',
                                'font-size': '14px',  # Increased font size
                                'text-wrap': 'wrap',
                                'width': '40px',      # Reduced node size
                                'height': '40px',     # Reduced node size
                                'padding': '8px',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'text-outline-width': 1,
                                'text-outline-color': '#FFFFFF'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'width': 1
                            }
                        }
                    ]
                ),
                style={'width': '50%', 'display': 'inline-block'}
            )
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

        dcc.Slider(
            id='time-slider',
            min=0, max=total_steps - 1, step=1, value=0,
            marks=time_slider_marks
        ),

        html.Div(id='info-output', style={'marginTop': 20, 'fontSize': '16px'}),
        
        # Add comparison graph
        html.Div([
            html.H3("County Case Comparison - Actual vs Predicted", style={'textAlign': 'center'}),
            dcc.Graph(id='comparison-graph')
        ], style={'marginTop': 30})
    ])

    @app.callback(
        [Output('cytoscape-graph', 'elements'),
         Output('markers', 'children'),
         Output('comparison-graph', 'figure')],
        Input('time-slider', 'value')
    )
    def update_visualization(time_index):
        snapshot = snapshots[time_index]
        
        # Get the original data for this time step from the raw dataset
        time_date = time_labels[time_index].split(' ')
        month, year = time_date[0], int(time_date[1])
        
        # Filter raw data for this time step
        raw_data_filtered = raw_data[
            (raw_data['Date'].dt.month == datetime.datetime.strptime(month, "%B").month) & 
            (raw_data['Date'].dt.year == year)
        ]
        
        # If we have data for this time step, use it
        if not raw_data_filtered.empty:
            true_cases = raw_data_filtered[csv_counties].values[0]
        else:
            # Use a default or placeholder value
            true_cases = np.zeros(len(csv_counties))
            
        # Get predicted cases for this time step
        if time_index < len(predictions):
            predicted_cases = predictions[time_index]
        else:
            predicted_cases = np.zeros(len(csv_counties))
        
        # Create color mapping based on case counts
        max_cases = max(np.max(true_cases), 1)  # Prevent division by zero
        
        # Generate colors based on intensity (red for higher cases)
        def get_color(cases):
            intensity = min(cases / max_cases, 1.0)
            r = int(255 * intensity)
            g = int(100 * (1 - intensity))
            b = int(100 * (1 - intensity))
            return f'rgb({r},{g},{b})'
        
        # Create network graph nodes
        nodes = []
        markers = []
        
        for i, county in enumerate(csv_counties):
            # Get true case count from raw data
            true_case_count = int(true_cases[i])
            predicted_case_count = int(predicted_cases[i])
            color = get_color(true_case_count)
            
            # Create cytoscape node
            nodes.append({
                'data': {
                    'id': county,
                    'label': f'{county}\n{true_case_count}\nPred: {predicted_case_count}',
                    'color': color,
                    'true_cases': true_case_count,
                    'predicted_cases': predicted_case_count
                },
                'position': positions[county]
            })
            
            # Create map marker
            lat, lon = county_coordinates[county]
            markers.append(
                dl.Marker(
                    position=[lat, lon],
                    children=[
                        dl.Tooltip(f"{county}: {true_case_count} cases (Pred: {predicted_case_count})")
                    ]
                )
            )
        
        # Create edges based on geographical proximity
        edges = []
        for i, county1 in enumerate(csv_counties):
            for j, county2 in enumerate(csv_counties):
                if i < j:  # Avoid duplicate edges
                    lat1, lon1 = county_coordinates[county1]
                    lat2, lon2 = county_coordinates[county2]
                    # Calculate distance
                    distance = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
                    # Connect nodes if they're close enough
                    if distance < 1.0:  # Approximately 100km
                        edges.append({
                            'data': {
                                'source': county1,
                                'target': county2
                            }
                        })
        
        # Create comparison graph
        comparison_figure = {
            'data': [
                {
                    'x': csv_counties,
                    'y': true_cases,
                    'type': 'bar',
                    'name': 'Actual Cases',
                    'marker': {'color': 'rgb(255, 50, 50)'}
                },
                {
                    'x': csv_counties,
                    'y': predicted_cases,
                    'type': 'bar',
                    'name': 'Predicted Cases',
                    'marker': {'color': 'rgb(50, 50, 255)'}
                }
            ],
            'layout': {
                'title': f'County Cases for {time_labels[time_index]}',
                'xaxis': {'title': 'County', 'tickangle': 45},
                'yaxis': {'title': 'Number of Cases'},
                'barmode': 'group',
                'legend': {'x': 0, 'y': 1.1, 'orientation': 'h'},
                'margin': {'l': 50, 'r': 50, 'b': 100, 't': 50}
            }
        }
        
        return nodes + edges, markers, comparison_figure

    @app.callback(
        Output('info-output', 'children'),
        [Input('cytoscape-graph', 'tapNodeData'),
         Input('time-slider', 'value')]
    )
    def display_click_info(node_data, time_index):
        current_time = time_labels.get(time_index, "Unknown")

        if node_data:
            county = node_data['id']
            true_cases = node_data['true_cases']
            predicted_cases = node_data['predicted_cases']
            error = abs(true_cases - predicted_cases)
            error_percent = (error / (true_cases + 1)) * 100  # Adding 1 to avoid division by zero
            
            return (f"Current time: {current_time} | County: {county} | "
                    f"Actual cases: {true_cases} | Predicted cases: {predicted_cases} | "
                    f"Error: {error:.1f} cases ({error_percent:.1f}%)")

        return f"Current time: {current_time} | Click on a county to see case details."

    return app

# Convert time steps to real dates
def generate_time_labels(start_year=2004, num_steps=500):
    start_date = datetime.date(start_year, 1, 1)
    dates = [start_date + datetime.timedelta(weeks=i) for i in range(num_steps)]
    return {i: date.strftime('%B %Y') for i, date in enumerate(dates)}

# Main Function
def main():
    # Load the dataset
    raw_data = load_raw_data()
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    
    # Create the dashboard with dataset snapshots
    app = create_dashboard(dataset, raw_data)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()