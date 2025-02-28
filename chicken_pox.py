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

# Step 7: Build Interactive Dashboard
def create_dashboard(predictions_over_time, dataset, max_cases):
    app = dash.Dash(__name__)

    total_steps = len(predictions_over_time)
    tick_spacing = max(total_steps // 12, 1)
    slider_marks = {i: f'Time {i}' if i % tick_spacing == 0 else '' for i in range(total_steps)}

    snapshot = dataset[0]  # Use the first snapshot for positions
    num_nodes = snapshot.x.shape[0]

    # Precompute node positions based on county coordinates
    positions = {}
    for county, (lat, lon) in county_coordinates.items():
        positions[county] = normalize_coordinates(lat, lon)

    app.layout = html.Div([  
        html.H1("Temporal GNN Visualization"),
        cyto.Cytoscape(
            id='cytoscape-graph',
            elements=[],
            layout={'name': 'preset'},  
            style={'width': '100%', 'height': '600px'},
            stylesheet=[
                {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '10px'}},  
                {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'triangle'}}  
            ]
        ),
        dcc.Slider(
            id='time-slider',
            min=0,
            max=total_steps - 1,
            step=1,
            value=0,
            marks=slider_marks
        ),
        html.Div(id='info-output', style={'marginTop': 20, 'fontSize': '16px'}),  
    ])

    @app.callback(
        Output('cytoscape-graph', 'elements'),
        Input('time-slider', 'value')
    )
    def update_graph(time_index):
        snapshot = dataset[time_index]
        node_values = np.clip((snapshot.x.numpy() * max_cases), 0, None).astype(int)

        # Create nodes with positions
        nodes = [
            {
                'data': {'id': county, 'label': f'{county}: {node_values[i, 0]} cases'},
                'position': positions[county]  
            }
            for i, county in enumerate(county_coordinates.keys())
        ]

        edges = [
            {
                'data': {
                    'source': list(county_coordinates.keys())[src],
                    'target': list(county_coordinates.keys())[dst],
                    'weight': w if snapshot.edge_attr is not None else 1  
                }
            }
            for src, dst, w in zip(
                snapshot.edge_index[0].numpy(),
                snapshot.edge_index[1].numpy(),
                snapshot.edge_attr.numpy() if snapshot.edge_attr is not None else [1] * len(snapshot.edge_index[0])
            )
            if src < len(county_coordinates) and dst < len(county_coordinates)  
        ]

        return nodes + edges

    @app.callback(
        Output('info-output', 'children'),
        Input('cytoscape-graph', 'tapNodeData'),
        Input('cytoscape-graph', 'tapEdgeData')
    )
    def display_click_info(node_data, edge_data):
        if node_data:
            return f"Node clicked: {node_data['id']}, Label: {node_data['label']}"
        elif edge_data:
            return f"Edge clicked: Source {edge_data['source']} → Target {edge_data['target']}, Weight: {edge_data['weight']:.2f}"
        return "Click on a node or edge to see more details."

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
