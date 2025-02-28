import torch
import torch_geometric_temporal
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
import dash_leaflet as dl
import datetime

# Dictionary of Hungarian counties with their approximate latitude and longitude
county_coordinates = {
    'Bács-Kiskun': (46.5935, 19.3547),
    'Baranya': (45.9910, 18.2232),
    'Békés': (46.7639, 21.0845),
    'Borsod-Abaúj-Zemplén': (48.2286, 20.6180),
    'Csongrád': (46.4167, 20.2500),
    'Fejér': (47.1130, 18.4367),
    'Győr-Moson-Sopron': (47.6849, 17.2610),
    'Hajdú-Bihar': (47.5160, 21.6500),
    'Heves': (47.8500, 20.0833),
    'Jász-Nagykun-Szolnok': (47.1667, 20.4167),
    'Komárom-Esztergom': (47.6833, 18.3333),
    'Nógrád': (48.0000, 19.5000),
    'Pest': (47.5000, 19.3333),
    'Somogy': (46.5833, 17.6667),
    'Szabolcs-Szatmár-Bereg': (47.9000, 22.0000),
    'Tolna': (46.5000, 18.5000),
    'Vas': (47.0833, 16.5667),
    'Veszprém': (47.1000, 17.9000),
    'Zala': (46.8333, 16.8333),
    'Budapest': (47.497913, 19.040236),
}

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

# Step 1: Create the Temporal GNN Model
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=32):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h = self.linear(h2)
        return h

# Step 2: Load data
def load_data():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    return dataset

# Step 3: Train the model
def train_model(dataset, epochs=50):
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    model = TemporalGNN(node_features=4)  # Assuming 4 features per node
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    for epoch in range(epochs):
        for snapshot in train_dataset:
            y_hat = model(snapshot.x, snapshot.edge_index)
            loss = torch.mean((y_hat-snapshot.y)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# Step 4: Save Predictions for Visualization
def get_predictions_over_time(model, dataset):
    model.eval()
    predictions_over_time = []
    
    with torch.no_grad():
        for snapshot in dataset:
            y_hat = model(snapshot.x, snapshot.edge_index).numpy()
            predictions_over_time.append(y_hat)
    
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

def create_dashboard(predictions_over_time, dataset, max_cases):
    app = dash.Dash(__name__)
    total_steps = len(predictions_over_time)
    time_labels = generate_time_labels(start_year=2004, num_steps=total_steps)

    # Pre-compute the positions for counties once
    positions = {county: normalize_coordinates(lat, lon) for county, (lat, lon) in county_coordinates.items()}

    # Space out the time slider ticks by displaying every 10th label
    step_size = max(1, total_steps // 10)
    time_slider_marks = {i: time_labels[i] for i in range(0, total_steps, step_size) if i < total_steps}

    app.layout = html.Div([ 
        html.H1("Temporal GNN Visualization", style={'textAlign': 'center'}),

        # Container for the map and the graph side by side
        html.Div([
            # Left side: Map (Leaflet)
            html.Div(
                dl.Map(
                    [
                        dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),  # Explicit tile layer
                        dl.Polygon(
                            positions=[list(reversed(coord)) for coord in county_coordinates.values()],
                            color='blue', fill=True, fillOpacity=0.3
                        )
                    ],
                    center=[37.7749, -122.4194],  # Set map center to a known location (e.g., San Francisco)
                    zoom=5,  # Adjust the zoom level
                    style={'width': '100%', 'height': '500px'}, id='map'
                ),
                style={'width': '50%', 'display': 'inline-block'}
            ),

            # Right side: Graph (Cytoscape)
            html.Div(
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    elements=[],  # We will update this dynamically
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '500px'}
                ),
                style={'width': '50%', 'display': 'inline-block'}
            )
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

        # Time slider to control the graph updates
        dcc.Slider(
            id='time-slider',
            min=0, max=total_steps - 1, step=1, value=0,
            marks=time_slider_marks  # Spaced out time ticks
        ),

        # Info section to display the current time and clicked node/edge info
        html.Div(id='info-output', style={'marginTop': 20, 'fontSize': '16px'})
    ])

    @app.callback(
        Output('cytoscape-graph', 'elements'),
        Input('time-slider', 'value')
    )
    def update_graph(time_index):
        snapshot = dataset[time_index]
        node_values = np.clip((snapshot.x.numpy() * max_cases), 0, None).astype(int)

        # Generate nodes with their predicted values
        nodes = [
            {'data': {'id': county, 'label': f'{county}: {node_values[i, 0]} cases'},
             'position': positions[county]}
            for i, county in enumerate(county_coordinates.keys())
        ]
        
        # Generate edges from the snapshot
        edges = [
            {'data': {'source': list(county_coordinates.keys())[src], 'target': list(county_coordinates.keys())[dst]}}
            for src, dst in zip(snapshot.edge_index[0].numpy(), snapshot.edge_index[1].numpy())
        ]
        
        # Combine nodes and edges into the elements list
        return nodes + edges

    @app.callback(
        Output('info-output', 'children'),
        Input('cytoscape-graph', 'tapNodeData'),
        Input('cytoscape-graph', 'tapEdgeData'),
        Input('time-slider', 'value')
    )
    def display_click_info(node_data, edge_data, time_index):
        # Display the current time based on the slider value
        current_time = time_labels.get(time_index, "Unknown")

        # If a node is clicked, display true vs predicted values
        if node_data:
            county = node_data['id']
            predicted_cases = int(node_data['label'].split(': ')[1].split(' ')[0])  # Extract predicted cases from label

            # Fetch the true value from the dataset (assuming dataset has true values)
            true_cases = dataset[time_index].y.numpy()[list(county_coordinates.keys()).index(county)]  # Get true value for the county
            
            return f"Current time: {current_time} | Node clicked: {county} | True cases: {true_cases} | Predicted cases: {predicted_cases}"

        # If an edge is clicked, show edge information
        elif edge_data:
            return f"Edge clicked: Source {edge_data['source']} → Target {edge_data['target']}"

        # Default message when nothing is clicked
        return f"Current time: {current_time} | Click on a node or edge to see more details."

    return app


# Main Function
def main():
    dataset = load_data()
    model = train_model(dataset)  
    predictions_over_time = get_predictions_over_time(model, dataset)  
    all_features = torch.cat([snapshot.x for snapshot in dataset], dim=0)  
    max_cases = all_features.max().item()  
    min_cases = all_features.min().item()  

    app = create_dashboard(predictions_over_time, dataset, max_cases=max_cases)  
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
