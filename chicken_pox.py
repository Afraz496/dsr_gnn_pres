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

def normalize_coordinates(lat, lon, width=500, height=500):
    # Normalize latitude and longitude to fit in a given width and height
    min_lat, max_lat = 45.74, 48.58
    min_lon, max_lon = 16.11, 22.90

    x = (lon - min_lon) / (max_lon - min_lon) * width
    y = (max_lat - lat) / (max_lat - min_lat) * height
    return {'x': x, 'y': y}

def extract_node_names(dataset):
    # Assuming each snapshot has a feature for each county in the dataset
    return [f"County {i}" for i in range(len(dataset[0].x))]

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

def inverse_transform(scaled_values, original_values):
    min_val, max_val = original_values.min(), original_values.max()
    return scaled_values * (max_val - min_val) + min_val

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=32):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        return self.linear(h2)

def train_model(dataset, epochs=50):
    train_dataset, _ = temporal_signal_split(dataset, train_ratio=0.8)
    model = TemporalGNN(node_features=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(epochs):
        for snapshot in train_dataset:
            y_hat = model(snapshot.x, snapshot.edge_index)
            loss = torch.mean((y_hat - snapshot.y)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def get_predictions_over_time(model, dataset, original_df):
    model.eval()
    predictions_over_time = []
    with torch.no_grad():
        for idx, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index).numpy().flatten()
            original_values = original_df.iloc[idx][csv_counties].values
            predictions_over_time.append(inverse_transform(y_hat, original_values))
    return predictions_over_time

# Step 5: Create a NetworkX graph for visualization
def create_graph(snapshot):
    G = nx.Graph()
    edge_index = snapshot.edge_index.numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(src, dst)
    return G

# Step 6: Prepare Dash Cytoscape Data
def prepare_cytoscape_data(G, predictions, time_step):
    nodes = [{'data': {'id': str(node), 'label': f"Node {node}", 'prediction': predictions[node]}} 
             for node in G.nodes]
    edges = [{'data': {'source': str(edge[0]), 'target': str(edge[1])}} for edge in G.edges]
    return nodes + edges

# Convert time steps to real dates
def generate_time_labels(start_year=2004, num_steps=500):
    start_date = datetime.date(start_year, 1, 1)
    dates = [start_date + datetime.timedelta(weeks=i) for i in range(num_steps)]
    return {i: date.strftime('%B %Y') for i, date in enumerate(dates)}

def create_dashboard(predictions_over_time, dataset, inverse_transformer):
    app = dash.Dash(__name__)
    total_steps = len(predictions_over_time)
    time_labels = generate_time_labels(start_year=2005, num_steps=total_steps)

    positions = {county: normalize_coordinates(lat, lon) for county, (lat, lon) in county_coordinates.items()}

    step_size = max(1, total_steps // 10)
    time_slider_marks = {i: time_labels[i] for i in range(0, total_steps, step_size) if i < total_steps}

    app.layout = html.Div([ 
        html.H1("Temporal GNN Visualization", style={'textAlign': 'center'}),

        html.Div([
            html.Div(
                dl.Map(
                    [
                        dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
                        dl.Polygon(
                            positions=[list(reversed(coord)) for coord in county_coordinates.values()],
                            color='blue', fill=True, fillOpacity=0.3
                        )
                    ],
                    center=[47.1625, 19.5033],
                    zoom=7,
                    style={'width': '100%', 'height': '500px'}, id='map'
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
                                'font-size': '8px',
                                'text-transform': 'capitalize',
                                'width': '50px',
                                'height': '50px',
                                'padding': '8px'
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

        html.Div(id='info-output', style={'marginTop': 20, 'fontSize': '16px'})
    ])

    @app.callback(
        Output('cytoscape-graph', 'elements'),
        Input('time-slider', 'value')
    )
    def update_graph(time_index):
        snapshot = dataset[time_index]
        node_values = inverse_transformer(snapshot.x.numpy()).astype(int)


        nodes = [
            {'data': {'id': county, 'label': f'{county}: {node_values[i, 0]} cases'},
             'position': positions[county]}
            for i, county in enumerate(county_coordinates.keys())
        ]

        edges = [
            {'data': {'source': list(county_coordinates.keys())[src], 'target': list(county_coordinates.keys())[dst]}}
            for src, dst in zip(snapshot.edge_index[0].numpy(), snapshot.edge_index[1].numpy())
        ]

        return nodes + edges

    @app.callback(
        Output('info-output', 'children'),
        Input('cytoscape-graph', 'tapNodeData'),
        Input('cytoscape-graph', 'elements'),
        Input('time-slider', 'value')
    )
    def display_click_info(node_data, elements, time_index):
        current_time = time_labels.get(time_index, "Unknown")

        if node_data:
            county = node_data['id']
            predicted_cases = int(node_data['label'].split(': ')[1].split(' ')[0])
            true_cases = int(inverse_transformer(dataset[time_index].y.numpy())[list(county_coordinates.keys()).index(county)])


            return f"Current time: {current_time} | Node clicked: {county} | True cases: {true_cases} | Predicted cases: {predicted_cases}"

        return f"Current time: {current_time} | Click on a node or edge to see more details."

    return app



# Main Function
def main():
    raw_data = load_raw_data()
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    model = train_model(dataset)
    predictions_over_time = get_predictions_over_time(model, dataset, raw_data)
    all_features = torch.cat([snapshot.x for snapshot in dataset], dim=0)  
    # https://archive.ics.uci.edu/dataset/580/hungarian+chickenpox+cases
    max_cases = 479
    min_cases = all_features.min().item()
    print(max_cases)  
    all_true_values = torch.cat([snapshot.y for snapshot in dataset], dim=0).numpy()

    # Define the inverse_transformer as a lambda for easy passing
    inverse_transformer = lambda scaled_values: inverse_transform(scaled_values, all_true_values)

    app = create_dashboard(predictions_over_time, dataset, inverse_transformer)  
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
