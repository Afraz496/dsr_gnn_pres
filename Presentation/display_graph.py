import torch
import torch_geometric_temporal
import networkx as nx
import folium
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from torch import nn, optim
from IPython.display import display, clear_output

# Load the ChickenPox dataset
loader = ChickenpoxDatasetLoader()
data = loader.get_dataset()
train_data, test_data = temporal_signal_split(data, train_ratio=0.8)

# Define a simple GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=train_data.features[0].shape[1], out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Train the model
model.train()
for epoch in range(20):
    total_loss = 0
    for snapshot in train_data:
        optimizer.zero_grad()
        output = model(snapshot.x, snapshot.edge_index)
        loss = loss_function(output.squeeze(), snapshot.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Sample Hungarian county coordinates (latitude, longitude)
county_coords = {
    "Budapest": (47.4979, 19.0402), "Pest": (47.5, 19.3), "Gyor": (47.6833, 17.6351),
    "Debrecen": (47.5316, 21.6273), "Miskolc": (48.1035, 20.7784), "Pecs": (46.0727, 18.2323),
    "Szeged": (46.253, 20.1486), "Kecskemet": (46.8964, 19.6897), "Nyiregyhaza": (47.9555, 21.7166)
}

# Build a NetworkX graph from the dataset
G = nx.Graph()
for i, (src, dst) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
    G.add_edge(src.item(), dst.item())

# Function to update the map for each time step
def update_map(time_step, model):
    clear_output(wait=True)
    m = folium.Map(location=[47.1625, 19.5033], zoom_start=7)
    
    # Get predictions from the model
    snapshot = test_data[time_step]
    model.eval()
    with torch.no_grad():
        predicted_cases = model(snapshot.x, snapshot.edge_index).squeeze().numpy()
    
    # Add updated county markers
    for i, (county, (lat, lon)) in enumerate(county_coords.items()):
        cases = predicted_cases[i] if i < len(predicted_cases) else 0
        folium.Marker(
            [lat, lon], 
            popup=f"{county}<br>Predicted Cases: {cases:.2f}"
        ).add_to(m)
    
    # Add graph edges
    for edge in G.edges():
        node1, node2 = edge
        if node1 < len(county_coords) and node2 < len(county_coords):
            coords1 = list(county_coords.values())[node1]
            coords2 = list(county_coords.values())[node2]
            folium.PolyLine([coords1, coords2], color="blue").add_to(m)
    
    display(m)

# Animate over time using predictions
for t in range(len(test_data.features)):
    update_map(t, model)
    time.sleep(1)
